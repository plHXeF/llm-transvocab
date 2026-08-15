"""Pure vocabulary-file loading, parsing, preview, and atomic persistence.

This module deliberately has no Streamlit or model-endpoint dependencies. The
UI can use :class:`VocabularyRepository` for both manual and batch imports, and
can display the structured issues returned by every parsing operation.
"""

from __future__ import annotations

import codecs
import csv
import hashlib
import io
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

from domain import Card, normalize_card_field


REQUIRED_COLUMNS = ("word", "pos", "meaning")
IssueSeverity = Literal["warning", "error"]
SourceFormat = Literal["csv", "text"]


@dataclass(frozen=True)
class VocabularyIssue:
    """A user-displayable problem found while reading or writing vocabulary."""

    code: str
    message: str
    severity: IssueSeverity = "error"
    line_number: int | None = None


@dataclass(frozen=True)
class ParseResult:
    """Result of parsing import content before comparing it with a library."""

    cards: tuple[Card, ...]
    issues: tuple[VocabularyIssue, ...]
    duplicates: tuple[Card, ...]
    source_format: SourceFormat
    encoding: str | None
    total_rows: int

    @property
    def valid_count(self) -> int:
        return len(self.cards)

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)


@dataclass(frozen=True)
class VocabularyLoadResult:
    """Current library contents plus non-fatal diagnostics."""

    cards: tuple[Card, ...]
    issues: tuple[VocabularyIssue, ...]
    duplicates: tuple[Card, ...]
    encoding: str | None
    total_rows: int
    fingerprint: str | None

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)


@dataclass(frozen=True)
class ImportPreview:
    """A side-effect-free preview suitable for rendering in a GUI."""

    parsed_cards: tuple[Card, ...]
    cards_to_add: tuple[Card, ...]
    duplicates_in_source: tuple[Card, ...]
    duplicates_in_library: tuple[Card, ...]
    issues: tuple[VocabularyIssue, ...]
    source_format: SourceFormat
    encoding: str | None
    total_rows: int
    base_fingerprint: str | None

    @property
    def can_commit(self) -> bool:
        return bool(self.cards_to_add)

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)


@dataclass(frozen=True)
class VocabularyWriteResult:
    """Outcome of an atomic replace or append operation."""

    success: bool
    cards: tuple[Card, ...]
    added_cards: tuple[Card, ...]
    skipped_duplicates: tuple[Card, ...]
    issues: tuple[VocabularyIssue, ...]
    fingerprint: str | None


def _decode_content(content: str | bytes) -> tuple[str | None, str | None, list[VocabularyIssue]]:
    if isinstance(content, str):
        return content, None, []

    if content.startswith(codecs.BOM_UTF8):
        try:
            return content.decode("utf-8-sig"), "utf-8-sig", []
        except UnicodeDecodeError as exc:
            return None, None, [
                VocabularyIssue(
                    code="decode_error",
                    message=f"文件带 UTF-8 BOM，但无法解码：{exc}",
                )
            ]

    utf8_error: UnicodeDecodeError | None = None
    try:
        return content.decode("utf-8"), "utf-8", []
    except UnicodeDecodeError as exc:
        utf8_error = exc

    try:
        return content.decode("gb18030"), "gb18030", []
    except UnicodeDecodeError as gb_error:
        return None, None, [
            VocabularyIssue(
                code="decode_error",
                message=(
                    "文件既不是有效的 UTF-8/UTF-8-SIG，也不是有效的 GB18030："
                    f"UTF-8={utf8_error}; GB18030={gb_error}"
                ),
            )
        ]


def _normalise_header(value: str) -> str:
    return normalize_card_field(value, casefold=True)


def _make_card(
    word: str,
    pos: str,
    meaning: str,
    *,
    line_number: int,
) -> tuple[Card | None, VocabularyIssue | None]:
    fields = tuple(normalize_card_field(value) for value in (word, pos, meaning))
    empty_fields = [name for name, value in zip(REQUIRED_COLUMNS, fields) if not value]
    if empty_fields:
        return None, VocabularyIssue(
            code="empty_field",
            message=f"字段不能为空：{', '.join(empty_fields)}",
            line_number=line_number,
        )

    try:
        return Card(word=fields[0], pos=fields[1], meaning=fields[2]), None
    except (TypeError, ValueError) as exc:
        return None, VocabularyIssue(
            code="invalid_card",
            message=f"词条无效：{exc}",
            line_number=line_number,
        )


def _header_indexes(
    row: Sequence[str],
    *,
    line_number: int,
) -> tuple[dict[str, int] | None, list[VocabularyIssue]]:
    normalised = [_normalise_header(value) for value in row]
    issues: list[VocabularyIssue] = []
    indexes: dict[str, int] = {}

    for column in REQUIRED_COLUMNS:
        matches = [index for index, value in enumerate(normalised) if value == column]
        if len(matches) > 1:
            issues.append(
                VocabularyIssue(
                    code="duplicate_header",
                    message=f"表头字段重复：{column}",
                    line_number=line_number,
                )
            )
        elif matches:
            indexes[column] = matches[0]

    missing = [column for column in REQUIRED_COLUMNS if column not in indexes]
    if missing:
        issues.append(
            VocabularyIssue(
                code="missing_columns",
                message=f"缺少必要列：{', '.join(missing)}",
                line_number=line_number,
            )
        )

    return (indexes if not issues else None), issues


def _deduplicate_parsed_cards(
    rows: Iterable[tuple[Card, int]],
) -> tuple[tuple[Card, ...], tuple[Card, ...], list[VocabularyIssue]]:
    cards: list[Card] = []
    duplicates: list[Card] = []
    issues: list[VocabularyIssue] = []
    seen_ids: set[str] = set()

    for card, line_number in rows:
        if card.card_id in seen_ids:
            duplicates.append(card)
            issues.append(
                VocabularyIssue(
                    code="duplicate_in_source",
                    message=f"导入内容中有重复词条：{card.word} / {card.pos} / {card.meaning}",
                    severity="warning",
                    line_number=line_number,
                )
            )
            continue
        seen_ids.add(card.card_id)
        cards.append(card)

    return tuple(cards), tuple(duplicates), issues


def _parse_delimited(
    text: str,
    *,
    delimiter: str,
    require_header: bool,
    source_format: SourceFormat,
    encoding: str | None,
    allow_comments: bool,
) -> ParseResult:
    issues: list[VocabularyIssue] = []
    parsed_rows: list[tuple[Card, int]] = []
    total_rows = 0
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter, strict=True)

    first_row: list[str] | None = None
    first_line = 1
    try:
        for row in reader:
            if not row or all(not value.strip() for value in row):
                continue
            if allow_comments and len(row) == 1 and row[0].lstrip().startswith("#"):
                continue
            first_row = row
            first_line = reader.line_num
            break
    except csv.Error as exc:
        issues.append(
            VocabularyIssue(
                code="csv_error",
                message=f"无法读取分隔文本：{exc}",
                line_number=reader.line_num or 1,
            )
        )

    if first_row is None:
        if not issues:
            issues.append(
                VocabularyIssue(
                    code="empty_input",
                    message="导入内容为空。",
                    severity="warning",
                )
            )
        return ParseResult((), tuple(issues), (), source_format, encoding, 0)

    header_values = {_normalise_header(value) for value in first_row}
    looks_like_header = bool(header_values.intersection(REQUIRED_COLUMNS))
    indexes: dict[str, int] | None = None
    expected_columns = 3

    if require_header or looks_like_header:
        indexes, header_issues = _header_indexes(first_row, line_number=first_line)
        issues.extend(header_issues)
        if indexes is None:
            return ParseResult((), tuple(issues), (), source_format, encoding, 0)
        expected_columns = len(first_row)
    else:
        indexes = {"word": 0, "pos": 1, "meaning": 2}

    def consume(row: Sequence[str], line_number: int) -> None:
        nonlocal total_rows
        if not row or all(not value.strip() for value in row):
            return
        if allow_comments and len(row) == 1 and row[0].lstrip().startswith("#"):
            return

        total_rows += 1
        if len(row) != expected_columns:
            issues.append(
                VocabularyIssue(
                    code="invalid_column_count",
                    message=f"应有 {expected_columns} 列，实际为 {len(row)} 列。",
                    line_number=line_number,
                )
            )
            return

        assert indexes is not None
        card, issue = _make_card(
            row[indexes["word"]],
            row[indexes["pos"]],
            row[indexes["meaning"]],
            line_number=line_number,
        )
        if issue is not None:
            issues.append(issue)
        elif card is not None:
            parsed_rows.append((card, line_number))

    if not (require_header or looks_like_header):
        consume(first_row, first_line)

    try:
        for row in reader:
            consume(row, reader.line_num)
    except csv.Error as exc:
        issues.append(
            VocabularyIssue(
                code="csv_error",
                message=f"CSV 第 {reader.line_num} 行附近格式错误：{exc}",
                line_number=reader.line_num,
            )
        )

    cards, duplicates, duplicate_issues = _deduplicate_parsed_cards(parsed_rows)
    issues.extend(duplicate_issues)
    return ParseResult(cards, tuple(issues), duplicates, source_format, encoding, total_rows)


def _parse_whitespace_text(text: str, *, encoding: str | None) -> ParseResult:
    issues: list[VocabularyIssue] = []
    parsed_rows: list[tuple[Card, int]] = []
    total_rows = 0
    records: list[tuple[int, list[str]]] = []

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        records.append((line_number, stripped.split(maxsplit=2)))

    if not records:
        return ParseResult(
            (),
            (VocabularyIssue("empty_input", "导入内容为空。", "warning"),),
            (),
            "text",
            encoding,
            0,
        )

    first_line, first_fields = records[0]
    header_values = {_normalise_header(value) for value in first_fields}
    looks_like_header = bool(header_values.intersection(REQUIRED_COLUMNS))
    start_index = 0
    indexes = {"word": 0, "pos": 1, "meaning": 2}
    if looks_like_header:
        header_indexes, header_issues = _header_indexes(first_fields, line_number=first_line)
        issues.extend(header_issues)
        if header_indexes is None:
            return ParseResult((), tuple(issues), (), "text", encoding, 0)
        indexes = header_indexes
        start_index = 1

    for line_number, fields in records[start_index:]:
        total_rows += 1
        if len(fields) != 3:
            issues.append(
                VocabularyIssue(
                    code="invalid_column_count",
                    message=f"应有 3 个字段，实际为 {len(fields)} 个字段。",
                    line_number=line_number,
                )
            )
            continue
        card, issue = _make_card(
            fields[indexes["word"]],
            fields[indexes["pos"]],
            fields[indexes["meaning"]],
            line_number=line_number,
        )
        if issue is not None:
            issues.append(issue)
        elif card is not None:
            parsed_rows.append((card, line_number))

    cards, duplicates, duplicate_issues = _deduplicate_parsed_cards(parsed_rows)
    issues.extend(duplicate_issues)
    return ParseResult(cards, tuple(issues), duplicates, "text", encoding, total_rows)


def parse_csv(content: str | bytes) -> ParseResult:
    """Parse a standard CSV containing a ``word,pos,meaning`` header."""

    text, encoding, decode_issues = _decode_content(content)
    if text is None:
        return ParseResult((), tuple(decode_issues), (), "csv", encoding, 0)
    result = _parse_delimited(
        text,
        delimiter=",",
        require_header=True,
        source_format="csv",
        encoding=encoding,
        allow_comments=False,
    )
    if not decode_issues:
        return result
    return ParseResult(
        result.cards,
        tuple(decode_issues) + result.issues,
        result.duplicates,
        result.source_format,
        result.encoding,
        result.total_rows,
    )


def parse_text(content: str | bytes) -> ParseResult:
    """Parse standard TXT/pasted rows.

    Tab, pipe, and comma-delimited rows are supported, with an optional header.
    When no delimiter is present, each line is split into ``word pos meaning``
    using at most two whitespace splits so meanings may contain spaces.
    """

    text, encoding, decode_issues = _decode_content(content)
    if text is None:
        return ParseResult((), tuple(decode_issues), (), "text", encoding, 0)

    sample_lines = [
        line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")
    ][:20]
    sample = "\n".join(sample_lines)
    delimiter: str | None = None
    for candidate in ("\t", "|"):
        if candidate in sample:
            delimiter = candidate
            break
    if delimiter is None and sample_lines:
        # An ASCII comma may legitimately occur inside the meaning of a
        # whitespace-delimited TXT row.  Treat comma as the row delimiter only
        # when the first non-comment line is actually a three-column CSV row.
        try:
            first_comma_row = next(csv.reader([sample_lines[0]], delimiter=",", strict=True))
        except (csv.Error, StopIteration):
            first_comma_row = []
        if len(first_comma_row) == 3:
            delimiter = ","

    if delimiter is None:
        result = _parse_whitespace_text(text, encoding=encoding)
    else:
        result = _parse_delimited(
            text,
            delimiter=delimiter,
            require_header=False,
            source_format="text",
            encoding=encoding,
            allow_comments=True,
        )

    if not decode_issues:
        return result
    return ParseResult(
        result.cards,
        tuple(decode_issues) + result.issues,
        result.duplicates,
        result.source_format,
        result.encoding,
        result.total_rows,
    )


def parse_import_content(
    content: str | bytes,
    *,
    source_name: str | None = None,
    source_format: Literal["csv", "txt", "text", "paste"] | None = None,
) -> ParseResult:
    """Parse uploaded bytes or pasted text using an explicit or inferred format."""

    selected_format = source_format
    if selected_format is None and source_name:
        selected_format = "csv" if Path(source_name).suffix.casefold() == ".csv" else "text"
    if selected_format is None:
        selected_format = "text"

    if selected_format == "csv":
        return parse_csv(content)
    if selected_format in {"txt", "text", "paste"}:
        return parse_text(content)
    raise ValueError(f"不支持的导入格式：{selected_format}")


class VocabularyRepository:
    """Repository for a CSV vocabulary file with atomic writes."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)

    @staticmethod
    def _fingerprint_bytes(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def fingerprint(self) -> str | None:
        """Return the content fingerprint, or ``None`` when no file exists."""

        try:
            return self._fingerprint_bytes(self.path.read_bytes())
        except (FileNotFoundError, OSError):
            return None

    def load(self) -> VocabularyLoadResult:
        try:
            raw = self.path.read_bytes()
        except FileNotFoundError:
            return VocabularyLoadResult(
                cards=(),
                issues=(
                    VocabularyIssue(
                        code="file_not_found",
                        message=f"词库文件不存在：{self.path}",
                        severity="warning",
                    ),
                ),
                duplicates=(),
                encoding=None,
                total_rows=0,
                fingerprint=None,
            )
        except OSError as exc:
            return VocabularyLoadResult(
                cards=(),
                issues=(VocabularyIssue("read_error", f"无法读取词库：{exc}"),),
                duplicates=(),
                encoding=None,
                total_rows=0,
                fingerprint=None,
            )

        parsed = parse_csv(raw)
        return VocabularyLoadResult(
            cards=parsed.cards,
            issues=parsed.issues,
            duplicates=parsed.duplicates,
            encoding=parsed.encoding,
            total_rows=parsed.total_rows,
            fingerprint=self._fingerprint_bytes(raw),
        )

    def preview_import(
        self,
        content: str | bytes,
        *,
        source_name: str | None = None,
        source_format: Literal["csv", "txt", "text", "paste"] | None = None,
    ) -> ImportPreview:
        parsed = parse_import_content(
            content,
            source_name=source_name,
            source_format=source_format,
        )
        current = self.load()
        existing_ids = {card.card_id for card in current.cards}
        cards_to_add: list[Card] = []
        duplicates_in_library: list[Card] = []
        issues = list(parsed.issues) + list(current.issues)

        for card in parsed.cards:
            if card.card_id in existing_ids:
                duplicates_in_library.append(card)
                issues.append(
                    VocabularyIssue(
                        code="duplicate_in_library",
                        message=f"词库中已存在：{card.word} / {card.pos} / {card.meaning}",
                        severity="warning",
                    )
                )
            else:
                existing_ids.add(card.card_id)
                cards_to_add.append(card)

        return ImportPreview(
            parsed_cards=parsed.cards,
            cards_to_add=tuple(cards_to_add),
            duplicates_in_source=parsed.duplicates,
            duplicates_in_library=tuple(duplicates_in_library),
            issues=tuple(issues),
            source_format=parsed.source_format,
            encoding=parsed.encoding,
            total_rows=parsed.total_rows,
            base_fingerprint=current.fingerprint,
        )

    def preview_file(self, path: str | os.PathLike[str]) -> ImportPreview:
        import_path = Path(path)
        try:
            content = import_path.read_bytes()
        except OSError as exc:
            current = self.load()
            source_format: SourceFormat = (
                "csv" if import_path.suffix.casefold() == ".csv" else "text"
            )
            return ImportPreview(
                parsed_cards=(),
                cards_to_add=(),
                duplicates_in_source=(),
                duplicates_in_library=(),
                issues=(VocabularyIssue("read_error", f"无法读取导入文件：{exc}"),),
                source_format=source_format,
                encoding=None,
                total_rows=0,
                base_fingerprint=current.fingerprint,
            )
        return self.preview_import(content, source_name=import_path.name)

    @staticmethod
    def _canonicalise_cards(
        cards: Iterable[Card],
    ) -> tuple[tuple[Card, ...], tuple[Card, ...], tuple[VocabularyIssue, ...]]:
        rows: list[tuple[Card, int]] = []
        issues: list[VocabularyIssue] = []
        for position, card in enumerate(cards, start=1):
            canonical, issue = _make_card(
                card.word,
                card.pos,
                card.meaning,
                line_number=position,
            )
            if issue is not None:
                issues.append(issue)
            elif canonical is not None:
                rows.append((canonical, position))
        unique, duplicates, duplicate_issues = _deduplicate_parsed_cards(rows)
        issues.extend(duplicate_issues)
        return unique, duplicates, tuple(issues)

    def _atomic_write(self, cards: Sequence[Card]) -> tuple[bool, VocabularyIssue | None]:
        temporary_path: Path | None = None
        try:
            # Directory creation is part of the write operation and may fail
            # for a read-only location or when a path component is a file.
            # Keep that failure inside the repository result boundary so the
            # Streamlit layer can render a diagnostic instead of crashing.
            self.path.parent.mkdir(parents=True, exist_ok=True)
            existing_mode = (
                stat.S_IMODE(self.path.stat().st_mode) if self.path.exists() else 0o644
            )
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                writer = csv.DictWriter(
                    temporary,
                    fieldnames=list(REQUIRED_COLUMNS),
                    lineterminator="\n",
                )
                writer.writeheader()
                for card in cards:
                    writer.writerow(
                        {"word": card.word, "pos": card.pos, "meaning": card.meaning}
                    )
                temporary.flush()
                os.fsync(temporary.fileno())

            os.chmod(temporary_path, existing_mode)
            os.replace(temporary_path, self.path)
            temporary_path = None
            return True, None
        except OSError as exc:
            return False, VocabularyIssue("write_error", f"无法写入词库：{exc}")
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def write_cards(self, cards: Iterable[Card]) -> VocabularyWriteResult:
        """Atomically replace the library with canonical, de-duplicated cards."""

        canonical, duplicates, validation_issues = self._canonicalise_cards(cards)
        success, write_issue = self._atomic_write(canonical)
        issues = list(validation_issues)
        if write_issue is not None:
            issues.append(write_issue)
        return VocabularyWriteResult(
            success=success,
            cards=canonical if success else self.load().cards,
            added_cards=canonical if success else (),
            skipped_duplicates=duplicates,
            issues=tuple(issues),
            fingerprint=self.fingerprint(),
        )

    def append_cards(self, cards: Iterable[Card]) -> VocabularyWriteResult:
        """Atomically append cards while preserving distinct senses.

        The current file is reloaded immediately before writing, so an old GUI
        preview cannot re-add a card that was imported in another rerun.
        """

        current = self.load()
        # Rewriting from ``current.cards`` would silently discard every invalid
        # source row that the tolerant loader skipped.  Refuse all appends when
        # the existing library has an error; callers can still display its valid
        # cards and diagnostics, but must repair the source before mutating it.
        if any(issue.severity == "error" for issue in current.issues):
            return VocabularyWriteResult(
                success=False,
                cards=current.cards,
                added_cards=(),
                skipped_duplicates=(),
                issues=tuple(current.issues),
                fingerprint=current.fingerprint,
            )

        incoming, incoming_duplicates, validation_issues = self._canonicalise_cards(cards)
        existing_ids = {card.card_id for card in current.cards}
        added: list[Card] = []
        existing_duplicates: list[Card] = []
        for card in incoming:
            if card.card_id in existing_ids:
                existing_duplicates.append(card)
            else:
                existing_ids.add(card.card_id)
                added.append(card)

        skipped = tuple(incoming_duplicates) + tuple(existing_duplicates)
        issues = list(current.issues) + list(validation_issues)
        for card in existing_duplicates:
            issues.append(
                VocabularyIssue(
                    code="duplicate_in_library",
                    message=f"词库中已存在：{card.word} / {card.pos} / {card.meaning}",
                    severity="warning",
                )
            )

        if not added:
            return VocabularyWriteResult(
                success=True,
                cards=current.cards,
                added_cards=(),
                skipped_duplicates=skipped,
                issues=tuple(issues),
                fingerprint=current.fingerprint,
            )

        merged = tuple(current.cards) + tuple(added)
        success, write_issue = self._atomic_write(merged)
        if write_issue is not None:
            issues.append(write_issue)
        return VocabularyWriteResult(
            success=success,
            cards=merged if success else current.cards,
            added_cards=tuple(added) if success else (),
            skipped_duplicates=skipped,
            issues=tuple(issues),
            fingerprint=self.fingerprint(),
        )

    def commit_preview(self, preview: ImportPreview) -> VocabularyWriteResult:
        """Commit previewed cards, safely rechecking a stale library."""

        stale_issue: VocabularyIssue | None = None
        if preview.base_fingerprint != self.fingerprint():
            stale_issue = VocabularyIssue(
                code="stale_preview",
                message="预览后词库已发生变化，已按最新词库重新去重。",
                severity="warning",
            )

        result = self.append_cards(preview.cards_to_add)
        if stale_issue is None:
            return result
        return VocabularyWriteResult(
            success=result.success,
            cards=result.cards,
            added_cards=result.added_cards,
            skipped_duplicates=result.skipped_duplicates,
            issues=(stale_issue,) + result.issues,
            fingerprint=result.fingerprint,
        )

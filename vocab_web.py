"""Streamlit UI for the vocabulary learner.

All model calls, persistence, scheduling, vocabulary parsing, and background
work live in dedicated modules.  This file only coordinates those services and
renders explicit user actions; merely opening a statistics/import page never
starts an LLM request.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import threading
import time
import uuid
from dataclasses import replace
from datetime import datetime
from typing import Any, Iterable, Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from app_settings import AppSettings, SettingsError, SettingsStore, default_app_settings
from domain import Card, Progress, ReviewEvent, utc_now
from learning_charts import baseline_figure, forgetting_figure
from learning_store import LearningStore
from llm_service import (
    EVALUATION_PROMPT_VERSION,
    LLMConfigurationError,
    LLMService,
    LLMServiceError,
    LLMSettings,
    PROMPT_VERSIONS,
    ReasoningEffort,
    SENTENCE_PROMPT_VERSION,
    SentenceDifficulty,
    SentenceResult,
    EvaluationResult,
    coerce_sentence_difficulty,
    redact_text,
)
from model_error_log import ModelErrorLog
from prefetch import PrefetchKey, PrefetchManager
from scheduler import build_batch, priority_for, retention_for
from vocabulary_repository import (
    ImportPreview,
    VocabularyIssue,
    VocabularyLoadResult,
    VocabularyRepository,
)


PAGES = ("开始学习", "学习数据", "词库管理")
EFFORT_LABELS = {
    ReasoningEffort.AUTO: "自动（由模型决定）",
    ReasoningEffort.DISABLED: "关闭",
    ReasoningEffort.MINIMAL: "最小",
    ReasoningEffort.LOW: "低",
    ReasoningEffort.MEDIUM: "中",
    ReasoningEffort.HIGH: "高",
    ReasoningEffort.MAX: "最大",
}
DIFFICULTY_LABELS = {
    SentenceDifficulty.JUNIOR_HIGH: "初中",
    SentenceDifficulty.GAOKAO_CET4: "高考/CET4",
    SentenceDifficulty.CET6_POSTGRAD: "CET6/考研",
    SentenceDifficulty.IELTS: "IELTS",
}
REGISTER_STYLES = {
    SentenceDifficulty.JUNIOR_HIGH: (
        "daily home life",
        "school life",
        "hobbies and friends",
        "shopping or travel",
    ),
    SentenceDifficulty.GAOKAO_CET4: (
        "campus life",
        "practical communication",
        "social and cultural topics",
        "work and travel",
    ),
    SentenceDifficulty.CET6_POSTGRAD: (
        "clear academic prose",
        "formal news analysis",
        "social or public issues",
        "science and technology",
        "descriptive general prose",
    ),
    SentenceDifficulty.IELTS: (
        "education",
        "environment and cities",
        "science and technology",
        "society and work",
        "international daily life",
    ),
}
MAX_IMPORT_BYTES = 2 * 1024 * 1024
MAX_IMPORT_LINES = 5_000
AI_IMPORT_CHUNK_LINES = 50
MEANING_REVEAL_SECONDS = 5


def _effort_options() -> tuple[ReasoningEffort, ...]:
    return (
        ReasoningEffort.AUTO,
        ReasoningEffort.DISABLED,
        ReasoningEffort.MINIMAL,
        ReasoningEffort.LOW,
        ReasoningEffort.MEDIUM,
        ReasoningEffort.HIGH,
        ReasoningEffort.MAX,
    )


def _register_for(
    card: Card,
    attempts: int,
    difficulty: SentenceDifficulty | str,
) -> str:
    level = coerce_sentence_difficulty(difficulty)
    styles = REGISTER_STYLES[level]
    digest = card.card_id.rsplit(":", 1)[-1]
    try:
        offset = int(digest[:8], 16)
    except ValueError:
        offset = 0
    return styles[(offset + max(0, attempts)) % len(styles)]


def _generate_sentence_worker(
    settings: LLMSettings,
    card: Card,
    register: str,
    difficulty: SentenceDifficulty | str,
) -> SentenceResult:
    """Background worker: deliberately contains no Streamlit access."""

    return LLMService(settings).generate_sentence(
        card,
        register=register,
        difficulty=difficulty,
    )


def _difficulty_recalibration_worker(db_path: str) -> None:
    """Background worker: recalibration is local and never touches Streamlit."""

    try:
        LearningStore(db_path).recalibrate_difficulty()
    except Exception:
        # Difficulty is only a bounded scheduling bonus. A failed refresh must
        # never interrupt learning; the next eligible answer will try again.
        return


def _schedule_difficulty_recalibration(store: LearningStore) -> None:
    if not store.difficulty_recalibration_due():
        return
    threading.Thread(
        target=_difficulty_recalibration_worker,
        args=(str(store.db_path),),
        name="vocab-difficulty-recalibration",
        daemon=True,
    ).start()


def _evaluator_profile(settings: LLMSettings, prompt_version: str) -> str:
    material = "\x00".join(
        (
            settings.resolved_base_url or "",
            settings.model,
            settings.reasoning_effort.value,
            prompt_version,
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _safe_ui_error(error: object, settings: LLMSettings | None = None) -> str:
    secrets = () if settings is None else (settings.api_key,)
    return redact_text(error, secrets=secrets)


def _record_model_error(
    operation: str,
    error: object,
    settings: LLMSettings | None = None,
    *,
    prompt_version: str | None = None,
    card: Card | None = None,
    category: str | None = None,
    status_code: int | None = None,
    request_id: str | None = None,
    error_type: str | None = None,
) -> None:
    """Best-effort diagnostics must never hide the original model failure."""

    try:
        ModelErrorLog(config.MODEL_ERROR_LOG_FILE).record(
            operation,
            error,
            settings,
            prompt_version=prompt_version,
            card_id=None if card is None else card.card_id,
            batch_id=str(st.session_state.get("batch_id") or "") or None,
            category=category,
            status_code=status_code,
            request_id=request_id,
            error_type=error_type,
        )
    except Exception as log_error:
        st.session_state.model_log_warning = (
            "模型错误发生了，但诊断日志无法写入：" + _safe_ui_error(log_error)
        )


def _load_initial_settings(store: SettingsStore) -> tuple[AppSettings, str | None]:
    try:
        return store.load_or_default(), None
    except SettingsError as error:
        return default_app_settings(), str(error)


def _initialize_session_state(initial_settings: AppSettings) -> None:
    defaults: dict[str, Any] = {
        "llm_settings": initial_settings.llm,
        "config_revision": 0,
        "remote_models": [],
        "remote_models_endpoint": "",
        "learning_active": False,
        "batch_complete": False,
        "batch_cards": [],
        "batch_id": "",
        "batch_vocabulary_fingerprint": "",
        "current_index": 0,
        "current_word_data": None,
        "current_generation_error": None,
        "evaluation_result": None,
        "evaluation_error": None,
        "meaning_revealed": False,
        "meaning_visible": False,
        "meaning_reveal_deadline": None,
        "batch_results": [],
        "batch_size": 20,
        "sentence_difficulty": initial_settings.sentence_difficulty.value,
        "applied_sentence_difficulty": initial_settings.sentence_difficulty.value,
        "import_rows": [],
        "import_rejected": [],
        "import_issues": [],
        "import_preview_revision": 0,
        "import_flash": None,
        "settings_flash": None,
        "data_flash": None,
        "model_log_flash": None,
        "model_log_warning": None,
        "settings_warning": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if not isinstance(st.session_state.remote_models, list):
        st.session_state.remote_models = []
        st.session_state.remote_models_endpoint = ""
    if "prefetch_manager" not in st.session_state:
        st.session_state.prefetch_manager = PrefetchManager()


def _cancel_prefetch() -> None:
    manager = st.session_state.get("prefetch_manager")
    if isinstance(manager, PrefetchManager):
        manager.cancel()


def _clear_batch_state(*, completed: bool = False) -> None:
    _cancel_prefetch()
    st.session_state.learning_active = False
    st.session_state.batch_complete = completed
    st.session_state.current_word_data = None
    st.session_state.current_generation_error = None
    st.session_state.evaluation_result = None
    st.session_state.evaluation_error = None
    st.session_state.meaning_revealed = False
    st.session_state.meaning_visible = False
    st.session_state.meaning_reveal_deadline = None
    if not completed:
        st.session_state.batch_cards = []
        st.session_state.batch_id = ""
        st.session_state.batch_vocabulary_fingerprint = ""
        st.session_state.current_index = 0
        st.session_state.batch_results = []


def _invalidate_model_work() -> None:
    _cancel_prefetch()
    if st.session_state.learning_active and st.session_state.evaluation_result is None:
        st.session_state.current_word_data = None
        st.session_state.current_generation_error = None
        st.session_state.evaluation_error = None


def _prefetch_key(card: Card) -> PrefetchKey:
    return PrefetchKey(
        card_id=card.card_id,
        config_revision=int(st.session_state.config_revision),
        batch_id=str(st.session_state.batch_id),
        prompt_version=SENTENCE_PROMPT_VERSION,
    )


def _start_batch(
    cards: Sequence[Card],
    progress_by_id: dict[str, Progress],
    batch_size: int,
    vocabulary_fingerprint: str | None,
) -> None:
    scheduled = build_batch(
        cards,
        progress_by_id,
        batch_size=batch_size,
        min_review_fraction=0.25,
    )
    st.session_state.batch_cards = [item.card for item in scheduled]
    st.session_state.batch_id = uuid.uuid4().hex
    st.session_state.batch_vocabulary_fingerprint = vocabulary_fingerprint or ""
    st.session_state.current_index = 0
    st.session_state.current_word_data = None
    st.session_state.current_generation_error = None
    st.session_state.evaluation_result = None
    st.session_state.evaluation_error = None
    st.session_state.meaning_revealed = False
    st.session_state.meaning_visible = False
    st.session_state.meaning_reveal_deadline = None
    st.session_state.batch_results = []
    st.session_state.batch_complete = False
    st.session_state.learning_active = bool(scheduled)
    _cancel_prefetch()


def _advance_card() -> None:
    st.session_state.current_index += 1
    st.session_state.current_word_data = None
    st.session_state.current_generation_error = None
    st.session_state.evaluation_result = None
    st.session_state.evaluation_error = None
    st.session_state.meaning_revealed = False
    st.session_state.meaning_visible = False
    st.session_state.meaning_reveal_deadline = None
    if st.session_state.current_index >= len(st.session_state.batch_cards):
        _clear_batch_state(completed=True)


def _prepare_current_card(store: LearningStore) -> None:
    if (
        st.session_state.current_word_data is not None
        or st.session_state.current_generation_error is not None
    ):
        return
    cards: list[Card] = st.session_state.batch_cards
    index = int(st.session_state.current_index)
    if index >= len(cards):
        _clear_batch_state(completed=True)
        return

    card = cards[index]
    progress = store.get_progress(card.card_id)
    difficulty = coerce_sentence_difficulty(st.session_state.sentence_difficulty)
    register = _register_for(
        card,
        0 if progress is None else progress.attempts,
        difficulty,
    )
    key = _prefetch_key(card)
    manager: PrefetchManager = st.session_state.prefetch_manager
    used_prefetch = manager.matches(key)
    try:
        with st.spinner(f"正在准备 {card.word} 的例句…"):
            if used_prefetch:
                sentence = manager.consume(key)
            else:
                sentence = _generate_sentence_worker(
                    st.session_state.llm_settings,
                    card,
                    register,
                    difficulty,
                )
        st.session_state.current_word_data = sentence
        st.session_state.current_generation_error = None
    except Exception as error:
        _record_model_error(
            "sentence_prefetch" if used_prefetch else "sentence_generation",
            error,
            st.session_state.llm_settings,
            prompt_version=SENTENCE_PROMPT_VERSION,
            card=card,
        )
        st.session_state.current_generation_error = _safe_ui_error(
            error, st.session_state.llm_settings
        )
        # Refresh once so the newly persisted diagnostic is visible in the
        # already-rendered sidebar. The guard above prevents an automatic retry.
        st.rerun()


def _schedule_next_prefetch(store: LearningStore) -> None:
    if not st.session_state.learning_active:
        return
    cards: list[Card] = st.session_state.batch_cards
    next_index = int(st.session_state.current_index) + 1
    if next_index >= len(cards):
        return
    next_card = cards[next_index]
    progress = store.get_progress(next_card.card_id)
    difficulty = coerce_sentence_difficulty(st.session_state.sentence_difficulty)
    register = _register_for(
        next_card,
        0 if progress is None else progress.attempts,
        difficulty,
    )
    key = _prefetch_key(next_card)
    manager: PrefetchManager = st.session_state.prefetch_manager
    if not manager.matches(key):
        manager.submit(
            key,
            _generate_sentence_worker,
            st.session_state.llm_settings,
            next_card,
            register,
            difficulty,
        )


def _format_local_time(value: datetime | None) -> str:
    if value is None:
        return "—"
    return value.astimezone().strftime("%Y-%m-%d %H:%M")


def _show_vocabulary_issues(
    issues: Iterable[VocabularyIssue], *, limit: int = 12
) -> None:
    issue_list = list(issues)
    for issue in issue_list[:limit]:
        location = f"（第 {issue.line_number} 行）" if issue.line_number else ""
        message = f"{issue.message}{location}"
        if issue.severity == "error":
            st.error(message)
        else:
            st.warning(message)
    if len(issue_list) > limit:
        st.caption(f"另有 {len(issue_list) - limit} 条诊断未展开。")


def _ensure_model_settings_draft() -> None:
    current: LLMSettings = st.session_state.llm_settings
    defaults = {
        "settings_base_url": current.resolved_base_url or config.DEFAULT_BASE_URL,
        "settings_new_api_key": "",
        "settings_key_action": "keep",
        "settings_effort": current.reasoning_effort.value,
        "settings_selected_model": "",
        "settings_manual_model": "",
    }
    if st.session_state.pop("settings_clear_new_key", False):
        st.session_state.settings_new_api_key = ""
        st.session_state.settings_key_action = "keep"
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _endpoint_draft_changed() -> None:
    """Clear endpoint-bound draft state when the URL changes."""

    st.session_state.settings_new_api_key = ""
    st.session_state.settings_key_action = "keep"
    st.session_state.settings_selected_model = ""
    st.session_state.settings_manual_model = ""
    st.session_state.remote_models = []
    st.session_state.remote_models_endpoint = ""


def _render_model_settings(settings_store: SettingsStore) -> None:
    current: LLMSettings = st.session_state.llm_settings
    _ensure_model_settings_draft()
    with st.expander("模型设置", expanded=False):
        base_url = st.text_input(
            "OpenAI-compatible Base URL",
            key="settings_base_url",
            on_change=_endpoint_draft_changed,
            placeholder="https://example.com/v1",
            help="模型和 Key 都按完整 Base URL 隔离保存。",
        )

        try:
            key_source = settings_store.api_key_source(base_url or None)
        except SettingsError as error:
            key_source = "none"
            st.warning(_safe_ui_error(error, current))
        source_labels = {
            "local": "已保存本地 Key",
            "environment": "使用环境变量 Key",
            "none": "未配置 Key",
        }
        st.caption(f"凭据状态：{source_labels[key_source]}")

        key_action = st.radio(
            "API Key 操作",
            ("keep", "replace", "delete"),
            format_func={
                "keep": "保留当前来源",
                "replace": "设置/替换本地 Key",
                "delete": "删除本地 Key",
            }.get,
            key="settings_key_action",
            horizontal=True,
            help="删除本地 Key 后会自动回退到对应环境变量。",
        )

        api_key = ""
        if key_action == "replace":
            api_key = st.text_input(
                "新的 API Key",
                type="password",
                key="settings_new_api_key",
                help="只用于搜索和本次替换；应用后立即从界面清空。",
            )

        effort_values = _effort_options()
        effort_value = st.selectbox(
            "全局思考强度",
            [item.value for item in effort_values],
            format_func=lambda value: EFFORT_LABELS[ReasoningEffort(value)],
            key="settings_effort",
            help="端点或模型不支持某档时，请改为“自动”以省略该参数。",
        )

        col_search, col_test = st.columns(2)
        if col_search.button("搜索可用模型", type="primary", width="stretch"):
            try:
                if key_action == "delete":
                    raise LLMConfigurationError("请先应用 Key 删除操作，再搜索模型")
                search_key = (
                    api_key
                    if key_action == "replace"
                    else settings_store.load_api_key(base_url or None)
                )
                if key_action == "replace" and not search_key.strip():
                    raise LLMConfigurationError("请输入新的 API Key")
                search_settings = LLMSettings(
                    base_url=base_url or None,
                    model="",
                    api_key=search_key,
                    reasoning_effort=effort_value,
                )
                with st.spinner("正在从 /models 获取模型列表…"):
                    models = LLMService(search_settings).list_models()
                st.session_state.remote_models = models
                st.session_state.remote_models_endpoint = (
                    search_settings.resolved_base_url or ""
                )
                if current.model in models:
                    st.session_state.settings_selected_model = current.model
                else:
                    st.session_state.settings_selected_model = models[0] if models else ""
                if models:
                    st.session_state.settings_flash = f"已发现 {len(models)} 个模型。"
                else:
                    st.session_state.settings_flash = "端点返回了空模型列表，可手动填写。"
                st.rerun()
            except Exception as error:
                _record_model_error(
                    "model_list",
                    error,
                    search_settings if "search_settings" in locals() else current,
                )
                st.error(_safe_ui_error(error, current))

        if col_test.button("测试已应用设置", width="stretch"):
            try:
                with st.spinner("正在测试当前已应用设置…"):
                    result = LLMService(current).test_connection()
                message = f"{result.message}（{result.latency_ms} ms）"
                if result.ok:
                    st.success(message)
                else:
                    _record_model_error(
                        "connection_test",
                        result.message,
                        current,
                        category=result.error_category,
                        status_code=result.status_code,
                        request_id=result.request_id,
                        error_type="ConnectionTestFailure",
                    )
                    st.error(message)
            except Exception as error:
                _record_model_error("connection_test", error, current)
                st.error(_safe_ui_error(error, current))

        normalized_draft_url = (base_url or "").rstrip("/")
        models = (
            st.session_state.remote_models
            if st.session_state.remote_models_endpoint == normalized_draft_url
            else []
        )
        selected_model = ""
        if models:
            if st.session_state.settings_selected_model not in models:
                st.session_state.settings_selected_model = models[0]
            selected_model = st.selectbox(
                "搜索结果（输入文字可筛选）",
                models,
                key="settings_selected_model",
                help="点击后直接输入模型名片段即可搜索。",
            )
            st.caption(f"当前端点返回 {len(models)} 个模型。")
        else:
            st.info("请先搜索模型；如果端点不支持 /models，可在下面手动填写。")

        manual_model = st.text_input(
            "手动模型名称（可选）",
            key="settings_manual_model",
            placeholder="仅在模型列表不可用时填写",
        )
        applied = st.button("应用设置", type="primary", width="stretch")

        if applied:
            try:
                model = manual_model.strip() or selected_model.strip()
                if not model and normalized_draft_url == (
                    current.resolved_base_url or ""
                ):
                    model = current.model
                if not model:
                    raise LLMConfigurationError("请先搜索选择模型，或手动填写模型名称")
                if key_action == "replace" and not api_key.strip():
                    raise LLMConfigurationError("请输入新的 API Key")
                draft = LLMSettings(
                    base_url=base_url or None,
                    model=model,
                    api_key=api_key,
                    reasoning_effort=effort_value,
                )
                draft.validate_for_request()
                settings_store.save(
                    AppSettings(
                        llm=draft,
                        sentence_difficulty=st.session_state.sentence_difficulty,
                    ),
                    key_action=key_action,
                )
                effective_key = settings_store.load_api_key(draft.resolved_base_url)
                effective = LLMSettings(
                    base_url=draft.base_url,
                    model=draft.model,
                    api_key=effective_key,
                    reasoning_effort=draft.reasoning_effort,
                    timeout_seconds=draft.timeout_seconds,
                    max_retries=draft.max_retries,
                )
                st.session_state.llm_settings = effective
                st.session_state.config_revision += 1
                st.session_state.settings_clear_new_key = True
                st.session_state.settings_flash = "模型设置已应用并保存。"
                _invalidate_model_work()
                st.rerun()
            except (SettingsError, LLMConfigurationError, ValueError) as error:
                st.error(_safe_ui_error(error, current))

        st.caption(
            f"当前生效：{current.resolved_base_url or '未配置端点'} · "
            f"{current.model or '未选择模型'} · "
            f"{EFFORT_LABELS[current.reasoning_effort]}"
        )


def _sentence_difficulty_changed(settings_store: SettingsStore) -> None:
    previous = coerce_sentence_difficulty(
        st.session_state.applied_sentence_difficulty
    )
    try:
        selected = coerce_sentence_difficulty(
            st.session_state.sentence_difficulty
        )
        settings_store.save(
            AppSettings(
                llm=st.session_state.llm_settings,
                sentence_difficulty=selected,
            ),
            key_action="keep",
        )
    except (SettingsError, LLMConfigurationError, ValueError) as error:
        st.session_state.sentence_difficulty = previous.value
        st.session_state.settings_warning = _safe_ui_error(
            error, st.session_state.llm_settings
        )
        return
    st.session_state.applied_sentence_difficulty = selected.value
    st.session_state.config_revision += 1
    st.session_state.settings_flash = (
        f"出题难度已切换为 {DIFFICULTY_LABELS[selected]}。"
    )
    _invalidate_model_work()


def _render_reset_controls(store: LearningStore) -> None:
    with st.expander("数据清理", expanded=False):
        if st.session_state.pop("clear_confirm_data_reset", False):
            st.session_state.confirm_data_reset = False
        confirmed = st.checkbox(
            "我确认执行下面选择的清理操作",
            key="confirm_data_reset",
        )
        if st.button(
            "重置熟练度/遗忘权重",
            disabled=not confirmed,
            width="stretch",
        ):
            count = store.reset_progress()
            _clear_batch_state()
            st.session_state.data_flash = (
                f"已重置 {count} 个词条的熟练度、遗忘和个人难度状态，"
                "历史仍保留。"
            )
            st.session_state.clear_confirm_data_reset = True
            st.rerun()
        if st.button(
            "清空学习历史",
            disabled=not confirmed,
            width="stretch",
        ):
            count = store.clear_history()
            st.session_state.batch_results = []
            st.session_state.data_flash = f"已删除 {count} 条历史，当前熟练度仍保留。"
            st.session_state.clear_confirm_data_reset = True
            st.rerun()
        if st.button(
            "清空全部学习数据",
            disabled=not confirmed,
            width="stretch",
        ):
            counts = store.clear_all()
            _clear_batch_state()
            st.session_state.data_flash = (
                f"已删除 {counts['progress']} 个进度和 "
                f"{counts['review_events']} 条历史。"
            )
            st.session_state.clear_confirm_data_reset = True
            st.rerun()
        st.caption("以上操作都不会修改 vocabularies.csv 或模型设置。")


def _model_error_dataframe(entries: Sequence[Any]) -> pd.DataFrame:
    category_labels = {
        "authentication": "认证失败",
        "rate_limit": "请求限流",
        "timeout": "请求超时",
        "connection": "连接失败",
        "api_status": "API 状态错误",
        "response_validation": "返回格式无效",
        "configuration": "配置错误",
        "model_service": "模型服务错误",
        "unknown": "未知错误",
    }

    def optional(value: Any) -> str:
        return "—" if value is None or value == "" else str(value)

    rows = []
    for entry in reversed(entries[-20:]):
        rows.append(
            {
                "时间(UTC)": entry.timestamp.replace("T", " ")[:19],
                "环节": entry.operation,
                "类别": category_labels.get(entry.category, entry.category),
                "状态": optional(entry.status_code),
                "结束原因": optional(entry.finish_reason),
                "输出 token": optional(entry.completion_tokens),
                "思考 token": optional(entry.reasoning_tokens),
                "尝试": optional(entry.attempts),
                "耗时(ms)": optional(entry.latency_ms),
                "模型": optional(entry.model),
                "错误": entry.message,
            }
        )
    return pd.DataFrame(rows, dtype="string")


def _render_model_diagnostics(error_log: ModelErrorLog) -> None:
    with st.expander("模型错误诊断", expanded=False):
        if st.session_state.model_log_flash:
            st.success(st.session_state.model_log_flash)
            st.session_state.model_log_flash = None
        if st.session_state.model_log_warning:
            st.warning(st.session_state.model_log_warning)
            st.session_state.model_log_warning = None

        entries = error_log.list_entries()
        st.caption("日志已脱敏，不含 API Key、提示词或作答内容。")
        if not entries:
            st.info("暂无模型错误日志。")
            return

        st.metric("已保留错误", len(entries))
        st.dataframe(
            _model_error_dataframe(entries),
            hide_index=True,
            width="stretch",
            height=min(420, 78 + 35 * min(20, len(entries))),
        )
        st.download_button(
            "下载脱敏日志",
            data=error_log.export_jsonl(),
            file_name="model_errors.redacted.jsonl",
            mime="application/x-ndjson",
            width="stretch",
        )

        if st.session_state.pop("clear_confirm_model_errors", False):
            st.session_state.confirm_clear_model_errors = False
        confirmed = st.checkbox(
            "我确认清空模型错误日志",
            key="confirm_clear_model_errors",
        )
        if st.button(
            "清空错误日志",
            disabled=not confirmed,
            width="stretch",
        ):
            count = error_log.clear()
            st.session_state.model_log_flash = f"已清空 {count} 条模型错误日志。"
            st.session_state.clear_confirm_model_errors = True
            st.rerun()


def _render_sidebar(
    settings_store: SettingsStore,
    learning_store: LearningStore,
    error_log: ModelErrorLog,
) -> str:
    with st.sidebar:
        st.markdown("## 学习控制台")
        _render_model_settings(settings_store)

        st.selectbox(
            "出题难度",
            [item.value for item in SentenceDifficulty],
            format_func=lambda value: DIFFICULTY_LABELS[
                SentenceDifficulty(value)
            ],
            key="sentence_difficulty",
            on_change=_sentence_difficulty_changed,
            args=(settings_store,),
        )
        st.number_input(
            "每批词数",
            min_value=5,
            max_value=100,
            step=5,
            key="batch_size",
            help="已有足够复习词时，至少 25% 的名额会保留给它们。",
        )
        st.markdown("---")
        page = st.radio("导航", PAGES, key="page_navigation")
        st.markdown("---")
        _render_model_diagnostics(error_log)
        _render_reset_controls(learning_store)
        if os.getenv("VOCAB_DESKTOP_MODE") == "1":
            st.markdown("---")
            if st.button("退出应用", width="stretch"):
                os._exit(0)
    return page


def _render_batch_summary() -> None:
    results: list[dict[str, Any]] = st.session_state.batch_results
    st.subheader("本轮完成")
    if results:
        scores = [
            float(item["score"])
            for item in results
            if item["status"] == "answered"
        ]
        skipped = sum(item["status"] == "skipped" for item in results)
        col1, col2, col3 = st.columns(3)
        col1.metric("本轮词数", len(results))
        col2.metric(
            "已评分题平均分",
            "—" if not scores else f"{sum(scores) / len(scores):.1f}",
        )
        col3.metric("跳过", skipped)
    else:
        st.info("本轮没有产生学习记录。")
    if st.button("返回并准备新一轮", type="primary"):
        _clear_batch_state()
        st.rerun()


def _render_recent_batch_results() -> None:
    results: list[dict[str, Any]] = st.session_state.batch_results
    if not results:
        return
    st.markdown("---")
    st.subheader("本轮最近记录")
    for item in reversed(results[-5:]):
        result_label = (
            "已跳过"
            if item["status"] == "skipped"
            else f"{item['score']:.0f} 分"
        )
        with st.expander(
            f"{item['word']} · {item['pos']} · {result_label}"
        ):
            if item.get("sentence"):
                st.write(item["sentence"])
            if item["status"] == "skipped":
                st.caption("本题已主动跳过。")
            else:
                st.write(f"你的翻译：{item.get('user_translation', '')}")
                st.write(f"反馈：{item.get('feedback', '')}")


def _record_skip(store: LearningStore, card: Card) -> bool:
    sentence_result: SentenceResult | None = st.session_state.current_word_data
    try:
        store.record_review(
            card,
            None,
            status="skipped",
            sentence=None if sentence_result is None else sentence_result.english_sentence,
            reference_translation=(
                None if sentence_result is None else sentence_result.chinese_translation
            ),
            review_key=(
                f"{st.session_state.batch_id}:"
                f"{st.session_state.current_index}:{card.card_id}"
            ),
        )
    except Exception as error:
        st.error(
            "无法保存跳过记录，本题尚未前进："
            + _safe_ui_error(error, st.session_state.llm_settings)
        )
        return False
    st.session_state.batch_results.append(
        {
            "card_id": card.card_id,
            "word": card.word,
            "pos": card.pos,
            "meaning": card.meaning,
            "score": 0.0,
            "status": "skipped",
            "sentence": None if sentence_result is None else sentence_result.english_sentence,
        }
    )
    return True


@st.fragment(run_every=1)
def _render_meaning_countdown(card: Card, index: int) -> None:
    deadline = st.session_state.meaning_reveal_deadline
    remaining = 0 if deadline is None else math.ceil(deadline - time.monotonic())
    if remaining <= 0:
        st.session_state.meaning_visible = False
        st.session_state.meaning_reveal_deadline = None
        st.rerun()
        return

    st.info(card.meaning)
    st.caption(f"剩余 {remaining} 秒")
    if st.button(
        "收起释义",
        key=f"hide_meaning_{st.session_state.batch_id}_{index}",
        width="stretch",
    ):
        st.session_state.meaning_visible = False
        st.session_state.meaning_reveal_deadline = None
        st.rerun()


def _render_one_time_meaning(card: Card, index: int) -> None:
    if not st.session_state.meaning_revealed:
        if st.button(
            "查看释义",
            key=f"reveal_meaning_{st.session_state.batch_id}_{index}",
            width="stretch",
        ):
            st.session_state.meaning_revealed = True
            st.session_state.meaning_visible = True
            st.session_state.meaning_reveal_deadline = (
                time.monotonic() + MEANING_REVEAL_SECONDS
            )
            st.rerun()
        return

    if st.session_state.meaning_visible:
        _render_meaning_countdown(card, index)
    else:
        st.button(
            "释义已查看",
            key=f"meaning_locked_{st.session_state.batch_id}_{index}",
            disabled=True,
            width="stretch",
        )


def _render_learning_page(
    vocabulary: VocabularyLoadResult,
    store: LearningStore,
) -> None:
    st.header("开始学习")
    if vocabulary.issues:
        with st.expander("词库诊断", expanded=not vocabulary.cards):
            _show_vocabulary_issues(vocabulary.issues)

    if st.session_state.batch_complete:
        _render_batch_summary()
        return

    if not st.session_state.learning_active:
        if not vocabulary.cards:
            st.warning("当前没有可学习的有效词条，请先到“词库管理”导入。")
            return
        active_ids = [card.card_id for card in vocabulary.cards]
        progress = store.load_progress(active_ids)
        reviewed = sum(item.attempts > 0 for item in progress.values())
        col1, col2, col3 = st.columns(3)
        col1.metric("有效词条", len(vocabulary.cards))
        col2.metric("已学词条", reviewed)
        col3.metric("未学词条", max(0, len(vocabulary.cards) - reviewed))
        st.info(
            "系统会优先安排需要复习和个人学习中较难的词条。"
        )
        if st.button("开始本轮", type="primary"):
            try:
                st.session_state.llm_settings.validate_for_request()
                _start_batch(
                    list(vocabulary.cards),
                    progress,
                    int(st.session_state.batch_size),
                    vocabulary.fingerprint,
                )
                st.rerun()
            except LLMConfigurationError as error:
                st.error(str(error))
        return

    if (vocabulary.fingerprint or "") != str(
        st.session_state.batch_vocabulary_fingerprint
    ):
        st.warning("检测到词库已变动；本轮继续使用启动时快照，新词库从下一轮生效。")

    cards: list[Card] = st.session_state.batch_cards
    index = int(st.session_state.current_index)
    if index >= len(cards):
        _clear_batch_state(completed=True)
        st.rerun()
        return

    _prepare_current_card(store)
    card = cards[index]
    progress_value = (index + 1) / max(1, len(cards))
    st.progress(progress_value, text=f"进度：{index + 1}/{len(cards)}")

    if st.session_state.current_generation_error:
        st.error(f"例句生成失败：{st.session_state.current_generation_error}")
        col_retry, col_skip = st.columns(2)
        if col_retry.button("重试生成", type="primary", width="stretch"):
            st.session_state.current_generation_error = None
            st.rerun()
        if col_skip.button("跳过该词", width="stretch"):
            if _record_skip(store, card):
                _advance_card()
                st.rerun()
        return

    sentence_result: SentenceResult | None = st.session_state.current_word_data
    if sentence_result is None:
        return
    _schedule_next_prefetch(store)

    main_col, result_col = st.columns([3, 2])
    with main_col:
        st.subheader(card.word)
        st.caption(f"词性：{card.pos}")
        st.markdown("**英文例句**")
        st.info(sentence_result.english_sentence)

        if st.session_state.evaluation_result is None:
            _render_one_time_meaning(card, index)
            with st.form(f"translation_form_{st.session_state.batch_id}_{index}"):
                user_translation = st.text_area(
                    "请输入中文翻译",
                    height=120,
                    placeholder="输入后点击提交；Ctrl/Cmd+Enter 也可提交表单。",
                )
                col_skip, col_submit = st.columns(2)
                skip_pressed = col_skip.form_submit_button(
                    "跳过", width="stretch"
                )
                submit_pressed = col_submit.form_submit_button(
                    "提交翻译", type="primary", width="stretch"
                )

            if skip_pressed:
                if _record_skip(store, card):
                    _advance_card()
                    st.rerun()
            if submit_pressed:
                if not user_translation.strip():
                    st.warning("请先输入翻译，或选择跳过。")
                else:
                    try:
                        with st.spinner("正在评估翻译…"):
                            evaluation = LLMService(
                                st.session_state.llm_settings
                            ).evaluate_translation(
                                sentence_result.english_sentence,
                                sentence_result.chinese_translation,
                                user_translation,
                                card=card,
                            )
                        meaning_revealed = bool(
                            st.session_state.meaning_revealed
                        )
                        if meaning_revealed:
                            evaluation = replace(
                                evaluation,
                                target_error_weight=1.0,
                                attribution_confidence=1.0,
                            )
                        store.record_review(
                            card,
                            evaluation.score,
                            status="answered",
                            sentence=sentence_result.english_sentence,
                            reference_translation=sentence_result.chinese_translation,
                            user_translation=user_translation,
                            feedback=evaluation.feedback,
                            target_error_weight=evaluation.target_error_weight,
                            attribution_confidence=evaluation.attribution_confidence,
                            non_target_error_tags=evaluation.non_target_error_tags,
                            evaluator_profile=_evaluator_profile(
                                st.session_state.llm_settings,
                                evaluation.prompt_version,
                            ),
                            meaning_revealed=meaning_revealed,
                            review_key=(
                                f"{st.session_state.batch_id}:"
                                f"{index}:{card.card_id}"
                            ),
                        )
                        _schedule_difficulty_recalibration(store)
                        st.session_state.evaluation_result = evaluation
                        st.session_state.evaluation_error = None
                        st.session_state.batch_results.append(
                            {
                                "card_id": card.card_id,
                                "word": card.word,
                                "pos": card.pos,
                                "meaning": card.meaning,
                                "score": float(evaluation.score),
                                "status": "answered",
                                "sentence": sentence_result.english_sentence,
                                "reference_translation": sentence_result.chinese_translation,
                                "user_translation": user_translation,
                                "feedback": evaluation.feedback,
                                "target_error_weight": evaluation.target_error_weight,
                                "attribution_confidence": evaluation.attribution_confidence,
                                "meaning_revealed": meaning_revealed,
                            }
                        )
                        st.rerun()
                    except Exception as error:
                        _record_model_error(
                            "translation_evaluation",
                            error,
                            st.session_state.llm_settings,
                            prompt_version=EVALUATION_PROMPT_VERSION,
                            card=card,
                        )
                        st.session_state.evaluation_error = _safe_ui_error(
                            error, st.session_state.llm_settings
                        )
            if st.session_state.evaluation_error:
                st.error(
                    "评分失败，未写入成绩；你的输入仍保留，可再次提交。\n\n"
                    + st.session_state.evaluation_error
                )

    with result_col:
        st.subheader("评估结果")
        evaluation: EvaluationResult | None = st.session_state.evaluation_result
        if evaluation is None:
            st.info("提交翻译后在这里显示分数与反馈。")
        else:
            color = "green" if evaluation.score >= 80 else "orange" if evaluation.score >= 60 else "red"
            st.markdown(
                f"<div style='font-size:3rem;text-align:center;color:{color};font-weight:700'>"
                f"{evaluation.score}</div>",
                unsafe_allow_html=True,
            )
            st.caption("分数（0–100）")
            st.info(evaluation.feedback)
            if (
                not st.session_state.meaning_revealed
                and evaluation.attribution_confidence > 0
            ):
                st.caption(
                    "本次扣分归因于目标词："
                    f"{evaluation.target_error_weight:.0%}"
                    f"（归因置信度 {evaluation.attribution_confidence:.0%}）"
                )
            st.markdown("**参考翻译**")
            st.write(sentence_result.chinese_translation)
            if st.button("下一个单词", type="primary", width="stretch"):
                _advance_card()
                st.rerun()

    _render_recent_batch_results()


def _event_dataframe(events: Sequence[ReviewEvent]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "时间": event.reviewed_at.astimezone(),
                "单词": event.word,
                "词性": event.pos,
                "释义": event.meaning,
                "分数": event.score,
                "状态": "跳过" if event.skipped else "已评分",
                "目标词错误归因": (
                    None
                    if event.target_error_weight is None
                    else round(event.target_error_weight * 100, 1)
                ),
                "查看释义": "是" if event.meaning_revealed else "否",
                "反馈": event.feedback or "",
            }
            for event in events
        ]
    )


def _render_statistics_page(
    vocabulary: VocabularyLoadResult,
    store: LearningStore,
) -> None:
    st.header("学习数据")
    events = store.list_review_events()
    active_ids = [card.card_id for card in vocabulary.cards]
    progress = store.load_progress(active_ids)

    answered = [event for event in events if not event.skipped]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("累计练习", len(events))
    col2.metric("已评分", len(answered))
    col3.metric("已学词条", len(progress))
    col4.metric(
        "历史平均分",
        (
            "—"
            if not answered
            else f"{sum(event.score for event in answered) / len(answered):.1f}"
        ),
    )

    if not events:
        st.info("暂无学习历史。完成或跳过题目后会立即记录。")
    else:
        frame = _event_dataframe(events)
        if answered:
            answered_frame = _event_dataframe(answered)
            answered_frame["日期"] = answered_frame["时间"].dt.date.astype(str)
            daily = answered_frame.groupby("日期", as_index=False)["分数"].mean()

            st.subheader("每日已评分题平均分")
            figure = go.Figure(
                go.Scatter(
                    x=daily["日期"],
                    y=daily["分数"],
                    mode="lines+markers",
                    line={"color": "#2E86AB", "width": 3},
                )
            )
            figure.update_layout(
                xaxis_title="日期",
                yaxis_title="平均分",
                yaxis={"range": [0, 100]},
                template="plotly_white",
                height=320,
                margin={"l": 40, "r": 20, "t": 20, "b": 40},
            )
            st.plotly_chart(figure, width="stretch")

            st.subheader("分数分布")
            histogram = px.histogram(
                answered_frame,
                x="分数",
                nbins=20,
                color_discrete_sequence=["#2E86AB"],
            )
            histogram.update_layout(showlegend=False, height=320)
            st.plotly_chart(histogram, width="stretch")
        else:
            st.info("目前只有跳过记录，暂无可计算的翻译分数。")

        st.subheader("最近记录")
        recent = frame.tail(20).iloc[::-1].copy()
        recent["时间"] = recent["时间"].map(
            lambda value: value.strftime("%m-%d %H:%M")
        )
        st.dataframe(
            recent[["时间", "单词", "词性", "分数", "状态"]],
            hide_index=True,
            width="stretch",
        )

    st.markdown("---")
    st.subheader("当前掌握与复习优先级")
    difficulty_status = store.difficulty_status()
    if difficulty_status["trained"]:
        st.caption(
            "个人难度模型已校准："
            f"{difficulty_status['valid_samples']} 个有效归因样本，"
            f"个人目标词表现基线 {difficulty_status['baseline']:.1%}；"
            f"累计到 {difficulty_status['next_training_at']} 个样本时后台更新。"
        )
    else:
        st.caption(
            "个人难度模型正在收集样本："
            f"{difficulty_status['valid_samples']}/"
            f"{difficulty_status['next_training_at']}；"
            "单个词条至少需要 3 个可信归因样本。"
        )
    if not progress:
        st.info("暂无当前熟练度；重置权重后历史仍会保留，但这里会清空。")

    card_by_id = {card.card_id: card for card in vocabulary.cards}
    rows: list[dict[str, Any]] = []
    for card_id, item in progress.items():
        card = card_by_id.get(card_id)
        if card is None:
            continue
        rows.append(
            {
                "单词": card.word,
                "词性": card.pos,
                "释义": card.meaning,
                "熟练度": round(item.mastery * 100, 1),
                "预计保留率": round(retention_for(item) * 100, 1),
                "个人难度": (
                    f"观察中 {item.difficulty_samples}/3"
                    if item.difficulty_samples < 3
                    else f"{item.difficulty * 100:.1f}"
                ),
                "复习优先级": round(priority_for(item) * 100, 1),
                "练习次数": item.attempts,
                "上次复习": _format_local_time(item.last_reviewed_at),
            }
        )
    if rows:
        mastery_frame = pd.DataFrame(rows).sort_values(
            ["复习优先级", "单词"], ascending=[False, True]
        )
        st.dataframe(mastery_frame, hide_index=True, width="stretch")
    _render_learning_visualizations(card_by_id, progress, store)


def _render_learning_visualizations(
    cards: dict[str, Card], progress: dict[str, Progress], store: LearningStore,
) -> None:
    now = utc_now()
    with st.expander("遗忘曲线", expanded=False):
        eligible = [
            card_id for card_id, item in progress.items()
            if card_id in cards and item.attempts > 0 and item.last_reviewed_at is not None
        ]
        eligible.sort(key=lambda card_id: (-priority_for(progress[card_id], now=now), card_id))
        if not eligible:
            st.session_state.pop("forgetting_card", None)
            st.info("暂无可展示的已复习词条。完成练习后可查看预计遗忘曲线。")
        else:
            if st.session_state.get("forgetting_card") not in eligible:
                st.session_state.forgetting_card = eligible[0]
            selected = st.selectbox(
                "查看词条", eligible, key="forgetting_card",
                format_func=lambda card_id: (
                    f"{cards[card_id].word} · {cards[card_id].pos} · {cards[card_id].meaning}"
                ),
            )
            days = st.selectbox("预测天数", [7, 30, 90], index=1, key="forgetting_days")
            item = progress[selected]
            col1, col2, col3 = st.columns(3)
            col1.metric("当前保留率", f"{retention_for(item, now=now):.1%}")
            col2.metric("稳定性", f"{item.stability_days:.2f} 天")
            col3.markdown("上次复习")
            col3.write(_format_local_time(item.last_reviewed_at))
            st.plotly_chart(forgetting_figure(item, days, now=now), width="stretch")
            st.caption(
                "曲线是假设期间不复习的模型预测。保留率表示记忆随时间衰减的比例，"
                "与熟练度不同；实际复习会更新稳定性和曲线。"
            )

    with st.expander("个人基线", expanded=False):
        snapshot = store.baseline_visualization()
        status = snapshot["status"]
        col1, col2, col3 = st.columns(3)
        col1.metric("已校准个人基线", f"{status['baseline']:.1%}" if status["trained"] else "—")
        col2.metric("有效样本", status["valid_samples"])
        col3.metric("下次校准门槛", f"{status['next_training_at']} 个样本")
        if not status["trained"]:
            st.info(
                f"正在收集有效样本：{status['valid_samples']}/{status['next_training_at']}。"
                "完成首次校准后显示实际表现与当前模型预期的对比。"
            )
        elif not snapshot["samples"]:
            st.info("当前周期暂无有效样本可供展示。")
        else:
            st.plotly_chart(baseline_figure(snapshot["samples"]), width="stretch")
            st.caption(
                f"展示当前重置周期内最近 {len(snapshot['samples'])} 个有效样本，包含不同词库的练习。"
                "预期值统一使用最新已校准模型计算；参考线上方表示实际表现高于预期。"
            )
        st.caption(
            "个人基线是目标词表现指标，与整句翻译平均分不同。"
            "跳过、查看释义和低置信度归因不参与基线拟合；首次需 30 个有效样本，"
            "此后每增加 20 个样本后台更新。查看图表不会触发校准。"
        )


def _decode_upload(content: bytes) -> str:
    if len(content) > MAX_IMPORT_BYTES:
        raise ValueError("文件超过 2 MB，请拆分后导入。")
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("文件不是有效的 UTF-8、UTF-8-SIG 或 GB18030 文本。")


def _text_chunks(text: str) -> list[str]:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) > MAX_IMPORT_LINES:
        raise ValueError(f"一次最多导入 {MAX_IMPORT_LINES} 行，请拆分文件。")
    return [
        "\n".join(lines[index : index + AI_IMPORT_CHUNK_LINES])
        for index in range(0, len(lines), AI_IMPORT_CHUNK_LINES)
    ]


_IMPORT_WORD_SHAPE = re.compile(r"^[A-Za-z][A-Za-z0-9'’. /-]{0,119}$")
_IMPORT_POS_SHAPE = re.compile(r"^[A-Za-z][A-Za-z0-9 .,/&()+-]{0,79}$")


def _local_import_looks_structured(preview: ImportPreview) -> bool:
    """Reject accidental whitespace triples before bypassing AI cleanup."""

    if not preview.parsed_cards or preview.has_errors:
        return False
    return all(
        _IMPORT_WORD_SHAPE.fullmatch(card.word) is not None
        and _IMPORT_POS_SHAPE.fullmatch(card.pos) is not None
        for card in preview.parsed_cards
    )


def _set_import_preview(
    cards: Iterable[Card],
    *,
    issues: Iterable[str] = (),
    rejected: Iterable[dict[str, str]] = (),
) -> None:
    st.session_state.import_rows = [
        {"word": card.word, "pos": card.pos, "meaning": card.meaning}
        for card in cards
    ]
    st.session_state.import_issues = list(issues)
    st.session_state.import_rejected = list(rejected)
    st.session_state.import_preview_revision += 1


def _ai_parse_text(text: str, settings: LLMSettings) -> tuple[list[Card], list[dict[str, str]]]:
    chunks = _text_chunks(text)
    if not chunks:
        raise ValueError("没有可解析的非空文本。")
    service = LLMService(settings)
    cards: list[Card] = []
    rejected: list[dict[str, str]] = []
    progress_bar = st.progress(0, text="正在用模型整理词条…")
    for index, chunk in enumerate(chunks, start=1):
        try:
            result = service.normalize_vocab_batch(chunk)
            cards.extend(Card(item.word, item.pos, item.meaning) for item in result.items)
            rejected.extend(
                {"source": item.source, "reason": item.reason}
                for item in result.rejected
            )
        except Exception as error:
            _record_model_error(
                "vocabulary_import",
                error,
                settings,
                prompt_version=PROMPT_VERSIONS["vocabulary_import"],
            )
            rejected.append(
                {
                    "source": f"第 {index} 批（{len(chunk.splitlines())} 行）",
                    "reason": _safe_ui_error(error, settings),
                }
            )
        progress_bar.progress(index / len(chunks), text=f"已处理 {index}/{len(chunks)} 批")
    progress_bar.empty()
    return cards, rejected


def _render_import_preview(repository: VocabularyRepository) -> None:
    rows: list[dict[str, str]] = st.session_state.import_rows
    has_diagnostics = bool(
        st.session_state.import_issues or st.session_state.import_rejected
    )
    if not rows and not has_diagnostics:
        return
    st.markdown("---")
    st.subheader("导入预览")
    if st.session_state.import_issues:
        with st.expander("解析诊断", expanded=True):
            for issue in st.session_state.import_issues:
                st.warning(issue)
    if st.session_state.import_rejected:
        with st.expander(
            f"未接受项目（{len(st.session_state.import_rejected)}）",
            expanded=False,
        ):
            st.dataframe(
                pd.DataFrame(st.session_state.import_rejected),
                hide_index=True,
                width="stretch",
            )

    if not rows:
        st.info("本次没有可导入的有效词条；原词库未发生变化。")
        if st.button("关闭解析结果", width="stretch"):
            _set_import_preview(())
            st.rerun()
        return

    editor_key = f"import_editor_{st.session_state.import_preview_revision}"
    edited = st.data_editor(
        pd.DataFrame(rows, columns=["word", "pos", "meaning"]),
        hide_index=True,
        num_rows="dynamic",
        width="stretch",
        key=editor_key,
        column_config={
            "word": st.column_config.TextColumn("单词", required=True),
            "pos": st.column_config.TextColumn("词性", required=True),
            "meaning": st.column_config.TextColumn("中文释义", required=True),
        },
    )

    valid: list[Card] = []
    invalid_count = 0

    def clean_cell(value: object) -> str:
        if value is None:
            return ""
        try:
            if bool(pd.isna(value)):
                return ""
        except (TypeError, ValueError):
            pass
        return str(value).strip()

    for raw in edited.to_dict("records"):
        word = clean_cell(raw.get("word"))
        pos = clean_cell(raw.get("pos"))
        meaning = clean_cell(raw.get("meaning"))
        if not word and not pos and not meaning:
            continue
        if not word or not pos or not meaning:
            invalid_count += 1
            continue
        valid.append(Card(word, pos, meaning))

    existing_ids = {card.card_id for card in repository.load().cards}
    seen: set[str] = set()
    source_duplicates = 0
    library_duplicates = 0
    candidates: list[Card] = []
    for card in valid:
        if card.card_id in seen:
            source_duplicates += 1
        elif card.card_id in existing_ids:
            library_duplicates += 1
        else:
            seen.add(card.card_id)
            candidates.append(card)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("有效行", len(valid))
    col2.metric("将新增", len(candidates))
    col3.metric("词库重复", library_duplicates)
    col4.metric("本批重复/缺字段", source_duplicates + invalid_count)

    confirm = st.checkbox(
        "我已检查预览内容",
        key=f"confirm_import_preview_{st.session_state.import_preview_revision}",
    )
    col_commit, col_cancel = st.columns(2)
    if col_commit.button(
        "确认批量导入",
        type="primary",
        disabled=not confirm or not candidates,
        width="stretch",
    ):
        result = repository.append_cards(valid)
        if result.success:
            st.session_state.import_flash = (
                f"成功新增 {len(result.added_cards)} 条；"
                f"跳过 {len(result.skipped_duplicates)} 条精确重复。"
            )
            _set_import_preview(())
            st.rerun()
        else:
            _show_vocabulary_issues(result.issues)
    if col_cancel.button("取消预览", width="stretch"):
        _set_import_preview(())
        st.rerun()


def _render_vocabulary_page(
    repository: VocabularyRepository,
    vocabulary: VocabularyLoadResult,
) -> None:
    st.header("词库管理")
    if st.session_state.import_flash:
        st.success(st.session_state.import_flash)
        st.session_state.import_flash = None

    unique_words = len({card.word.casefold() for card in vocabulary.cards})
    col1, col2, col3 = st.columns(3)
    col1.metric("有效词条", len(vocabulary.cards))
    col2.metric("不同拼写", unique_words)
    col3.metric("编码", vocabulary.encoding or "—")
    if vocabulary.issues:
        with st.expander("词库诊断", expanded=vocabulary.has_errors):
            _show_vocabulary_issues(vocabulary.issues)

    manual_tab, batch_tab = st.tabs(["手动添加", "批量导入"])
    with manual_tab:
        with st.form("manual_vocabulary_form", clear_on_submit=False):
            word = st.text_input("单词")
            pos = st.text_input("词性", placeholder="例如 n / v / adj")
            meaning = st.text_input("中文释义")
            manual_add, manual_ai = st.columns(2)
            add_pressed = manual_add.form_submit_button(
                "准备预览", width="stretch"
            )
            ai_pressed = manual_ai.form_submit_button(
                "AI 补全并预览", type="primary", width="stretch"
            )
        if add_pressed:
            if not word.strip() or not pos.strip() or not meaning.strip():
                st.error("本地添加需要完整填写单词、词性和释义。")
            else:
                _set_import_preview((Card(word.strip(), pos.strip(), meaning.strip()),))
                st.rerun()
        if ai_pressed:
            if not word.strip():
                st.error("至少需要填写单词。")
            else:
                raw = f"word={word}\npos={pos or '(missing)'}\nmeaning={meaning or '(missing)'}"
                try:
                    cards, rejected = _ai_parse_text(raw, st.session_state.llm_settings)
                    _set_import_preview(cards, rejected=rejected)
                    st.rerun()
                except Exception as error:
                    if isinstance(error, LLMServiceError):
                        _record_model_error(
                            "vocabulary_import",
                            error,
                            st.session_state.llm_settings,
                            prompt_version=PROMPT_VERSIONS["vocabulary_import"],
                        )
                    st.error(_safe_ui_error(error, st.session_state.llm_settings))

    with batch_tab:
        uploaded = st.file_uploader(
            "上传 CSV 或 TXT",
            type=["csv", "txt"],
            accept_multiple_files=False,
        )
        pasted = st.text_area(
            "或粘贴文本",
            height=180,
            placeholder="可粘贴标准 word,pos,meaning，也可粘贴散乱单词清单交给 AI 整理。",
        )
        use_ai = st.checkbox(
            "非标准文本使用当前模型解析并补全",
            value=True,
        )
        if st.button("解析并生成预览", type="primary"):
            if uploaded is not None and pasted.strip():
                st.error("上传文件和粘贴文本请二选一。")
            elif uploaded is None and not pasted.strip():
                st.error("请上传文件或粘贴文本。")
            else:
                try:
                    if uploaded is not None:
                        raw_bytes = uploaded.getvalue()
                        raw_text = _decode_upload(raw_bytes)
                        is_standard_csv = uploaded.name.casefold().endswith(".csv")
                        source_name = uploaded.name
                        local_content: str | bytes = raw_bytes
                    else:
                        raw_text = pasted
                        if len(raw_text.encode("utf-8")) > MAX_IMPORT_BYTES:
                            raise ValueError("粘贴内容超过 2 MB，请拆分后导入。")
                        is_standard_csv = False
                        source_name = "pasted.txt"
                        local_content = raw_text

                    preview = repository.preview_import(
                        local_content,
                        source_name=source_name,
                        source_format="csv" if is_standard_csv else "text",
                    )
                    issue_text = [issue.message for issue in preview.issues]
                    local_parse_is_complete = _local_import_looks_structured(preview)
                    if is_standard_csv or local_parse_is_complete or not use_ai:
                        _set_import_preview(preview.parsed_cards, issues=issue_text)
                    else:
                        cards, rejected = _ai_parse_text(
                            raw_text, st.session_state.llm_settings
                        )
                        _set_import_preview(
                            cards,
                            issues=issue_text,
                            rejected=rejected,
                        )
                    st.rerun()
                except Exception as error:
                    if isinstance(error, LLMServiceError):
                        _record_model_error(
                            "vocabulary_import",
                            error,
                            st.session_state.llm_settings,
                            prompt_version=PROMPT_VERSIONS["vocabulary_import"],
                        )
                    st.error(_safe_ui_error(error, st.session_state.llm_settings))

    _render_import_preview(repository)

    st.markdown("---")
    with st.expander("浏览当前词库", expanded=False):
        st.dataframe(
            pd.DataFrame(
                [
                    {"word": card.word, "pos": card.pos, "meaning": card.meaning}
                    for card in vocabulary.cards
                ]
            ),
            hide_index=True,
            width="stretch",
            height=420,
        )


def main() -> None:
    st.set_page_config(
        page_title=config.PAGE_TITLE,
        page_icon=config.PAGE_ICON,
        layout=config.PAGE_LAYOUT,
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .main-title {color:#2E86AB;font-weight:700;margin-bottom:.25rem}
        .stButton > button {border-radius:8px}
        section[data-testid="stSidebar"][aria-expanded="true"] {
            min-width: 390px !important;
            max-width: 390px !important;
        }
        section[data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 390px !important;
        }
        @media (max-width: 700px) {
            section[data-testid="stSidebar"][aria-expanded="true"],
            section[data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
                min-width: min(390px, 88vw) !important;
                max-width: 88vw !important;
                width: 88vw !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"<h1 class='main-title'>{config.PAGE_TITLE}</h1>", unsafe_allow_html=True)

    settings_store = SettingsStore()
    initial_settings, settings_error = _load_initial_settings(settings_store)
    _initialize_session_state(initial_settings)
    learning_store = LearningStore(config.LEARNING_DB_FILE)
    error_log = ModelErrorLog(config.MODEL_ERROR_LOG_FILE)
    repository = VocabularyRepository(config.VOCAB_FILE)
    vocabulary = repository.load()

    if settings_error:
        st.warning(f"本地模型设置无法读取，已使用环境默认值：{settings_error}")
    if learning_store.last_recovery_backup is not None:
        st.warning(
            "学习数据库损坏，已备份并自动重建："
            f"{learning_store.last_recovery_backup.name}"
        )
    if st.session_state.settings_flash:
        st.success(st.session_state.settings_flash)
        st.session_state.settings_flash = None
    if st.session_state.settings_warning:
        st.warning(st.session_state.settings_warning)
        st.session_state.settings_warning = None

    page = _render_sidebar(settings_store, learning_store, error_log)
    if st.session_state.data_flash:
        st.success(st.session_state.data_flash)
        st.session_state.data_flash = None
    if page == "开始学习":
        _render_learning_page(vocabulary, learning_store)
    elif page == "学习数据":
        _render_statistics_page(vocabulary, learning_store)
    else:
        _render_vocabulary_page(repository, vocabulary)

if __name__ == "__main__":
    main()

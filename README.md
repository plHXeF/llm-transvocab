# LLM TransVocab

一个面向“熟词僻义”和长期复习、以翻译练习为核心的本地 Streamlit 词汇学习应用。它调用任意 OpenAI-compatible 模型生成指定义项的英文例句、评估英译中答案，再结合做题人的能力基础、熟练度与艾宾浩斯式时间衰减智能安排复习。

![学习首页](assets/screenshots/learning-home.png)

## 功能概览

- 需要 OpenAI-compatible 端点，不绑定特定供应商
- GUI 切换 Base URL、API Key、模型与全局思考强度
- 四档例句难度：初中、高考/CET4、CET6/考研和 IELTS
- 默认 20 词短批次，复习词按时间权重均匀穿插
- 同一拼写的不同词性、不同义项分别学习和记忆
- 当前题作答时，单线程后台预生成下一题
- AI 评分允许合理意译，并提供中文改进建议
- AI 将失分归因到目标词或上下文，学习个人“难记词”倾向
- 例句允许目标词的标准屈折变化，不要求机械使用词典原形
- CSV/TXT/粘贴文本批量导入，支持 AI 整理与可编辑预览
- SQLite 持久化个人能力基线、熟练度和逐题历史
- 脱敏错误日志记录结束原因、token 用量、耗时和请求 ID

## 快速开始

### 下载桌面版

不想安装 Python 时，可以从 [GitHub Releases](https://github.com/plHXeF/llm-transvocab/releases) 下载与电脑匹配的压缩包：

| 系统 | 下载文件 |
| --- | --- |
| Windows 10/11 64 位 | `llm-transvocab-windows-x64.zip` |
| Apple 芯片 Mac | `llm-transvocab-macos-arm64.zip` |
| Intel Mac | `llm-transvocab-macos-x64.zip` |

解压后直接运行 `LLM TransVocab`。程序会在本机启动服务并自动打开浏览器；使用完毕可在左侧点击“退出应用”。这些社区构建不包含代码签名：Windows 首次运行可能显示 SmartScreen，macOS 首次运行可能需要在 Finder 中右键应用并选择“打开”。

桌面版把设置、Key、学习历史、错误日志和可编辑词库放在用户数据目录，更新应用不会覆盖它们：

- macOS：`~/Library/Application Support/LLM TransVocab/`
- Windows：`%LOCALAPPDATA%\LLM TransVocab\`

压缩包旁的 `.sha256` 文件可用于核对下载完整性。

### 从源码运行

推荐 Python 3.11。本项目开发和测试使用 conda 环境：

```bash
conda create -n english python=3.11 -y
conda activate english
python -m pip install -r requirements.txt
streamlit run vocab_web.py
```

默认地址是 `http://localhost:8501`。首次打开后，先在左侧展开“模型设置”。

## 配置模型

应用使用 OpenAI Chat Completions 兼容接口。最少需要三个信息：

| 配置项 | 含义 | DeepSeek 示例 |
| --- | --- | --- |
| Base URL | OpenAI-compatible API 根地址 | `https://api.deepseek.com` |
| API Key | 该端点签发的访问密钥 | 在 DeepSeek 平台创建 |
| Model | `/models` 返回的模型 ID | `deepseek-v4-flash` |

![DeepSeek 模型配置示例](assets/screenshots/model-settings-deepseek.png)

以 DeepSeek 为例：在 [DeepSeek API Keys](https://platform.deepseek.com/api_keys) 创建 Key，将 Base URL 设为 `https://api.deepseek.com`，搜索模型后选择接口当前返回的 V4 模型。不要使用已经退役的 `deepseek-chat` 或 `deepseek-reasoner`；可用模型以 `/models` 的实时结果为准。

1. 填写服务提供的 Base URL，例如常见地址会以 `/v1` 结尾。
2. 设置该端点需要的 Key；不需要鉴权的本地端点可以留空。
3. 点击“搜索可用模型”并选择返回的模型。
4. 如果端点没有实现 `GET /models`，在“手动模型名称”中填写服务端要求的精确 ID。
5. “自动”会省略 `reasoning_effort`；其他档位会按 OpenAI-compatible 字段发送。如果模型不支持该字段，改为“自动”或“关闭”后重新测试。
6. 应用设置并执行连接测试。

完整功能至少需要端点兼容：

- `POST /chat/completions`
- `response_format={"type":"json_object"}`，或能够稳定返回 JSON 对象
- 可选的 `GET /models`
- 可选的 `reasoning_effort`

也可以用环境变量提供启动默认值。只有本地没有已保存 Key 时，环境变量才会作为后备：

```bash
export VOCAB_BASE_URL="https://api.example.com/v1"
export VOCAB_API_KEY="your-key"
export VOCAB_MODEL="your-model-id"
export VOCAB_REASONING_EFFORT="disabled"
```

## 出题难度

左侧“出题难度”会立即保存并应用于当前尚未提交的题目和后续预缓存。默认档位为“CET6/考研”。难度只控制例句的上下文词汇、句法和题材；无论目标词本身属于什么等级，模型都必须使用指定词义。

| 档位 | 例句设计 |
| --- | --- |
| 初中 | 8–15 词，日常和校园场景，基础词汇与简单句法 |
| 高考/CET4 | 12–22 词，常见社会和实用语境，允许常见从句 |
| CET6/考研 | 18–30 词，学术、社会和新闻分析语境，包含清晰的复杂关系 |
| IELTS | 18–32 词，教育、环境、科技等 Academic/General 常见主题 |

## 学习与复习机制

每次有效评分都会更新熟练度和稳定期。基础复习优先级为：

```text
retention = exp(-elapsed_days / stability_days)
priority  = 1 - mastery × retention
```

评分模型还会返回 `target_error_weight`：本次总扣分中，有多少源于不理解目标词的指定义项。系统据此计算目标词表现，并与该用户在相似熟练度、遗忘程度和练习次数下的预期表现比较。同一词条积累至少 3 个可信归因样本后才启用个人难度；难度最多为基础优先级增加 0.15，避免少量难词霸占批次。

每题提交前可以查看一次释义，释义会显示 5 秒，也可以提前收起，关闭后不能再次打开。该题仍由模型正常评分，不过系统会忽略模型给出的目标词错误归因，固定按 `target_error_weight=1`、归因置信度 `1` 保存。翻译分数照常进入历史，但这次辅助作答不参与个人平均能力拟合，并在该词的 difficulty 中作为一次目标词回忆失败。这样不会因为查看释义后译得流畅而把陌生词误判为容易。

个人校准模型只使用本机历史中的数值特征，不会再次调用 API：30 个有效归因样本后首次拟合，之后每新增 20 个样本在后台更新。评分模型切换信息只保存为不含 Key 的哈希标识。旧版历史没有归因字段，会被安全保留但不参与难度拟合。

优先级越高，越早进入后续批次。默认每批 20 词；已有足够复习词时，至少 25% 的名额来自已学词，并均匀分布在批次中。主动跳过按 0 分更新进度，但不作为个人难度样本；模型调用失败不写成绩。

![学习数据与遗忘权重](assets/screenshots/learning-data.png)


## 词库导入

默认词库是 UTF-8 CSV：

```csv
word,pos,meaning
abandon,v,放弃
abstract,adj,抽象的
```

“词库管理”页面支持：

- 手动添加完整词条
- 用当前模型补齐缺失的词性或释义
- 上传 UTF-8、UTF-8-SIG 或 GB18030 的 CSV/TXT
- 粘贴标准三列表格并在本地解析
- 将散乱文本按每批 50 行交给模型整理
- 在 `data_editor` 中修改预览后一次确认、原子写入
- 跳过完全相同的三字段记录，同时保留同词异义项

![AI 批量导入预览](assets/screenshots/vocabulary-import-preview.png)

标准 CSV/TXT 优先在本地解析，不消耗模型额度；勾选“非标准文本使用当前模型解析并补全”后，无法可靠识别为规范三列的数据会交给模型。模型返回的内容只进入预览，用户确认前不会修改词库。

### 熟词僻义导入建议

词条身份是规范化后的 `word + pos + meaning` SHA-256。同一拼写的不同词性或义项拥有各自的例句、熟练度和复习记录，因此它们可能出现在同一批次中。例如 `strain` 的“压力”和“菌株”会被当作两张不同卡片。只有三项完全相同的记录才会去重。

CSV 新增词会自动成为未学习词；删除词的旧进度会静默成为 orphan；修改词性或释义会生成新的词条 ID。当前批次使用开始时的词库快照，外部修改从下一批生效。

## AI请求的结构化输出与重试

造句、评分和词库整理都使用版本化提示词、JSON 模式和本地字段校验。为了兼容强制 reasoning 的模型：

- 造句和评分从 2000 output tokens 开始，重请求时依次使用 3000、4000、5000
- 批量导入从 20000 开始，重请求时依次使用 22000、24000、26000
- 空正文、`finish_reason=length`、无效 JSON 或字段校验失败时最多重试 3 次
- 非空但不合格的 JSON 会进入单独的修复提示；仍失败时向界面返回可重试错误

DeepSeek 官方也提示 JSON Output 偶尔可能返回空正文，并建议在提示词中提供 JSON 示例、设置合理的 `max_tokens`；本项目已经实现这些保护。[DeepSeek JSON Output 文档](https://api-docs.deepseek.com/guides/json_mode/)

## 本地数据与安全

| 路径 | 内容 | 是否应提交 |
| --- | --- | --- |
| `vocabularies.csv` | 词库 | 是 |
| `data/app_settings.json` | 非敏感模型设置 | 否 |
| `data/api_keys.json` | 按 Base URL 隔离的 Key | 否 |
| `data/learning.db` | 熟练度与作答历史 | 否 |
| `data/model_errors.jsonl` | 脱敏模型错误日志 | 否 |

Key 文件使用原子写入并设置为 `0600`。它仍是本机权限保护的明文文件，不是加密保险箱；多人共享机器时建议只使用环境变量或系统密钥管理工具。整个 `data/`、`.env` 与 Streamlit 本地凭据文件均已加入 `.gitignore`。

错误日志不会保存 API Key、请求头、提示词、模型原文、例句或用户译文；最多保留最近 500 条。侧栏可查看、下载或清空日志。

> 旧版本曾在 `config.py` 中包含明文 DeepSeek Key。升级不会自动撤销旧 Key；如果使用过该版本，请在 DeepSeek 控制台立即轮换。


## 数据清理

- “重置熟练度/遗忘权重”：清空熟练度、遗忘和个人难度状态，保留历史图表
- “清空学习历史”：删除逐题历史，保留当前熟练度
- “清空全部学习数据”：同时删除二者

这些操作均需二次确认，也不会修改 `vocabularies.csv` 或模型配置。

## 验证

测试使用标准库 `unittest` 和模拟 OpenAI 客户端，不消耗真实 API：

```bash
conda run -n english python -m py_compile *.py tests/*.py
conda run -n english python -m unittest discover -s tests -v
conda run -n english python -m pip check
```

测试覆盖模型参数与密钥脱敏、结构化输出修复、错误归因、个人难度校准、词库导入、SQLite 迁移、遗忘曲线、同词多义隔离、预缓存失效、Streamlit 页面状态流和数据清理。

## 项目结构

```text
vocab_web.py              Streamlit 页面和学习状态机
llm_service.py            OpenAI-compatible 请求、提示词与 JSON 校验
app_settings.py           本地设置和 Key 的安全读写
vocabulary_repository.py  词库解析、去重、预览和原子提交
learning_store.py         SQLite 进度、历史与迁移
scheduler.py              熟练度、遗忘曲线和批次选择
prefetch.py               下一题后台预缓存
model_error_log.py        脱敏模型错误日志
desktop_launcher.py       桌面包启动与浏览器打开
desktop_runtime.py        桌面版用户数据目录和首次初始化
packaging/                PyInstaller 构建配置
domain.py                 Card、Progress 等领域对象
tests/                    标准库 unittest 测试
```

## 许可证

本项目采用 [MIT License](LICENSE)。

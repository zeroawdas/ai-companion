# AI 伴侣记忆系统（ai-companion）

一个基于 **RAG 记忆** 的 AI 伴侣对话系统：不再用全量历史拼接撑上下文，而是把「值得长期记住的信息」抽成结构化记忆存入向量库，每轮对话按需检索相关记忆注入，实现低成本的长期记忆能力。

> 当前完成 **P1 + P2**：近期窗口 + 混合检索（BM25+向量）记忆 + 去重 + 人设注入 + 关系状态摘要 + 联网搜索。

## 项目简介

传统长对话受上下文窗口限制，聊天轮次越多，每轮携带的历史越重，token 成本随轮数快速上升。本项目的思路是：

- 为 AI 伴侣建立**独立记忆库**（FAISS 向量索引 + 结构化 JSON）。
- 每轮聊天用当前输入做**语义检索**，只取 top-k 相关记忆注入上下文。
- 同时保留**近期窗口**（最近若干轮原始对话）兜底语气与指代。
- 对话结束后，用 LLM **异步提取**新记忆写入记忆库。

### 技术选型

- **Python 3.12** + **Streamlit**（交互界面）
- **LangChain**（编排 + FAISS 向量库封装）
- **FAISS**（本地向量索引）+ **BM25**（关键词检索，混合召回）
- **DeepSeek**（对话生成 + 记忆提取）
- **本地 BGE 模型**（fastembed + bge-small-zh-v1.5 向量化，免费离线）
- **博查 Bocha**（联网搜索 API，国内直连，按需获取实时信息）

## 快速开始

### 环境要求

- Python 3.10+
- DeepSeek API Key（联网需额外配置博查 Bocha API Key，可选）

### 安装运行

```bash
# 1. 进入项目目录
cd ai-companion

# 2. 配置 .env（填写你的 API Key）
cp .env.example .env   # 如已存在则直接编辑

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动
streamlit run app.py
```

浏览器打开 `http://localhost:8501` 即可开始聊天。

`.env` 需包含：

```
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_API_BASE=https://api.deepseek.com
BOCHA_API_KEY=           # 可选，联网搜索用
```

## 使用说明

- 左侧边栏实时展示**当前记忆库**，可一键「清空对话」或「清空记忆库」。
- 聊天时系统会：检索相关记忆 → 组装（人设 + 记忆 + 近期窗口）→ 流式生成回复 → 提取新记忆写入。
- 人设定义在 `persona.py`，可自行修改角色名、性格与说话风格。
- 记忆库默认持久化在 `~/.ai-companion/memory_data`（英文路径，因 FAISS 不支持中文路径），可通过 `.env` 的 `MEMORY_DIR` 覆盖。
- 左侧边栏可打开「🌐 联网模式」，打开后小满会在需要时联网搜索实时信息（天气、新闻、日期等）。
- 首次启动会从国内镜像下载 BGE 模型（约 100MB），缓存到 `~/.ai-companion/models`。

## 项目结构

```
ai-companion/
├── app.py               # Streamlit 主程序（编排数据流）
├── config.py            # 配置（读 .env）
├── embeddings.py        # 本地 BGE embedding 封装
├── llm_client.py        # DeepSeek LLM 客户端
├── persona.py           # 人设与系统提示词
├── memory_store.py      # FAISS 记忆存储 + 检索
├── memory_extractor.py  # LLM 记忆提取
├── relationship_state.py # 关系状态摘要
├── bocha_search.py      # 博查联网搜索客户端
└── .env                 # API Key 配置
```

## 实施路线

- [x] **P1 MVP**：近期窗口 + 向量记忆检索 + 人设 + 记忆写入
- [x] **P2 记忆质量**：去重合并、混合检索（BM25+向量+RRF）、关系状态摘要
- [ ] **P3 体验增强**：三层记忆架构、指代消解、主动回忆、遗忘/压缩机制
- [ ] **P4 生产化**：评测集、成本监控、隐私删除权

## 许可证

MIT

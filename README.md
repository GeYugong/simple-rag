# simple-rag

一个最小可运行的 RAG 示例项目（当前为离线 TF-IDF 检索版本，不依赖外部模型下载）。

## 1. 项目目标

- 读取本地文档（`data/docs`）
- 文本切分（chunk）
- 构建检索索引（TF-IDF）
- 输入问题后检索 top-k 相关片段
- 展示“基于检索结果”的回答骨架

## 2. 环境要求

- Python 3.8+
- Windows PowerShell（其他终端也可）

## 3. 安装依赖

在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## 4. 准备文档

把你的知识库文档放到 `data/docs/`，支持：

- `.txt`
- `.md`

项目已提供示例文档，可直接运行。

## 5. 构建索引（Ingest）

```powershell
python src/ingest.py
```

成功后会在 `data/index/` 生成：

- `tfidf_matrix.npz`
- `tfidf_vocab.json`
- `tfidf_idf.npy`
- `meta.json`
- `index_params.json`

## 6. 运行问答（Retrieve + 结果拼接）

```powershell
python src/rag.py
```

然后输入问题，例如：

```text
RAG 为什么能减少胡说？
```

程序会输出 top-k 检索片段和分数。

## 7. 常见问题

### 7.1 报错：`Index not found. Run: python src/ingest.py`

说明你还没构建索引，先执行：

```powershell
python src/ingest.py
```

### 7.2 终端中文乱码

PowerShell 建议使用 UTF-8。临时设置：

```powershell
chcp 65001
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
```

### 7.3 我想接入大模型生成最终答案

当前 `src/rag.py` 中的 `naive_generate_answer` 只是演示文本拼接。你可以在这个函数里接入任意 LLM API，把检索到的 `contexts` 作为提示词上下文。

## 8. 项目结构

```text
simple-rag/
├─ data/
│  ├─ docs/          # 文档输入
│  └─ index/         # ingest 生成的索引文件
├─ src/
│  ├─ ingest.py      # 构建索引
│  └─ rag.py         # 检索与演示回答
└─ requirements.txt
```

## 9. 快速开始（最短路径）

```powershell
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
python src/ingest.py
python src/rag.py
```

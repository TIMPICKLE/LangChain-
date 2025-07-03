# LangChain 向量数据库与检索器

这个项目演示了如何使用 LangChain 0.3 创建一个基于本地嵌入模型的向量数据库和检索增强生成（RAG）系统，不依赖第三方在线服务。

## 项目概述

本项目包含以下核心功能：
- 使用本地模型生成文本嵌入（基于 sentence-transformers）
- 使用 FAISS 作为向量数据库存储引擎
- 连接自定义的 LLM API（基于内部部署的服务）
- 创建检索增强生成（RAG）系统
- 支持文档处理、分割和加载

## 环境要求

- Python 3.8+
- 所需的库已在 `requirements.txt` 文件中列出

## 安装步骤

1. 克隆或下载本项目到本地

2. 创建并激活虚拟环境（推荐）
   ```powershell
   # 创建虚拟环境
   python -m venv venv
   
   # 激活虚拟环境
   .\venv\Scripts\Activate.ps1
   ```

3. 安装依赖包
   ```powershell
   pip install -r requirements.txt
   ```

4. 配置 `.env` 文件（如需修改内部LLM服务配置）
   ```
   API_BASE_URL="http://your-api-url"
   API_MODEL_NAME="your-model-name"
   ```
   
## 使用方法

### 1. 添加示例数据

首先，添加一些示例数据到向量数据库：

```powershell
python main.py add --sample
```

### 2. 添加自定义文本

```powershell
python main.py add --text "这是一条测试文本，用于演示向量数据库的功能。" --source "测试数据"
```

### 3. 在向量数据库中搜索

```powershell
python main.py search "向量数据库是什么" --k 3
```

### 4. 使用RAG系统进行查询

```powershell
python main.py query "解释一下检索增强生成的工作原理"
```

### 5. 清空向量数据库

```powershell
python main.py clear
```

### 6. 测试LLM连接

```powershell
python main.py test-llm
```

## 加载文档

本项目支持加载不同类型的文档：

### 加载单个文本文件

```powershell
python load_documents.py load-text path/to/your/file.txt
```

### 加载PDF文件

```powershell
python load_documents.py load-pdf path/to/your/document.pdf
```

### 加载整个目录

```powershell
python load_documents.py load-dir path/to/your/directory --pattern "**/*.txt"
```

## 项目结构

```
vector_db_demo/
│
├── main.py                   # 主程序入口
├── custom_llm.py             # 自定义语言模型
├── local_embeddings.py       # 本地文本嵌入模型
├── vector_store.py           # 向量存储和检索
├── rag_system.py             # RAG系统
├── document_processor.py     # 文档处理工具
├── load_documents.py         # 文档加载示例
├── sample_data.py            # 示例数据
├── .env                      # 环境变量配置
└── requirements.txt          # 项目依赖
```

## 技术说明

1. **嵌入模型**：使用 `sentence-transformers` 本地生成文本嵌入，支持中文和多语言
2. **向量数据库**：使用 FAISS（Facebook AI Similarity Search）作为向量存储引擎
3. **语言模型**：使用内部部署的 LLM 服务
4. **文档处理**：支持文本和PDF文件加载，使用递归文本分割器进行文档分割

## 注意事项

- 首次运行时，会自动下载 sentence-transformers 模型，请确保有良好的网络连接
- 向量数据库存储在本地 `vector_db` 目录中，可以根据需要修改存储路径
- 对于大型文档，可以调整文档分割的参数（chunk_size 和 chunk_overlap）以获得更好的检索效果

## 国内快速下载huggingface（镜像）上的模型和数据
- 安装依赖
pip install -U huggingface_hub

- 设置环境变量

Linux

export HF_ENDPOINT=https://hf-mirror.com
Windows Powershell

$env:HF_ENDPOINT = "https://hf-mirror.com"

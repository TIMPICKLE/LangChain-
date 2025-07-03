"""
示例数据加载工具
用于加载示例数据到向量数据库
"""
import os
from typing import List, Dict, Any

def load_sample_data() -> List[Dict[str, Any]]:
    """加载示例文档数据"""
    documents = [
        {
            "text": "LangChain是一个用于构建基于大型语言模型应用的框架。它提供了各种工具和组件，可以帮助开发者轻松地将大型语言模型集成到各种应用中。",
            "metadata": {"source": "LangChain简介", "category": "概述"}
        },
        {
            "text": "向量数据库是一种专门用于存储和检索向量数据的数据库。在LLM应用中，向量数据库通常用于存储文档或其他信息的嵌入向量，以便进行高效的相似度搜索。",
            "metadata": {"source": "向量数据库介绍", "category": "技术组件"}
        },
        {
            "text": "检索增强生成（RAG）是一种将检索系统与生成模型结合的方法。该方法先通过检索系统找到与查询相关的文档或信息，然后将这些信息作为上下文提供给生成模型，以生成更准确、更相关的回答。",
            "metadata": {"source": "RAG技术简介", "category": "技术方法"}
        },
        {
            "text": "嵌入（Embeddings）是将文本、图像等转换为数值向量的过程。这些向量捕捉了原始内容的语义信息，使得可以通过计算向量之间的相似度来比较不同内容的语义相似性。",
            "metadata": {"source": "嵌入技术", "category": "基础概念"}
        },
        {
            "text": "LangChain的核心组件包括链（Chains）、代理（Agents）、记忆（Memory）以及检索器（Retrievers）等。这些组件可以灵活组合，构建出复杂的应用。",
            "metadata": {"source": "LangChain组件", "category": "框架结构"}
        },
        {
            "text": "FAISS（Facebook AI Similarity Search）是Facebook开发的一个高效的相似性搜索库。它专门用于在大型数据集中进行高效的相似性搜索和聚类。在LangChain中，FAISS常被用作向量存储的后端。",
            "metadata": {"source": "FAISS简介", "category": "技术组件"}
        },
        {
            "text": "提示工程（Prompt Engineering）是设计和优化提示以获得更好的语言模型响应的过程。一个好的提示可以显著提高模型输出的质量和相关性。",
            "metadata": {"source": "提示工程指南", "category": "最佳实践"}
        },
        {
            "text": "LangChain提供了多种向量存储选项，包括FAISS、Chroma、Pinecone等。选择适合自己需求的向量存储是构建高效检索系统的关键。",
            "metadata": {"source": "向量存储选项", "category": "技术选择"}
        },
        {
            "text": "在RAG系统中，文档分割（Document Splitting）是一个重要步骤。适当的分割策略可以提高检索的精确度和相关性。常见的分割方法包括按段落、按句子或固定长度进行分割。",
            "metadata": {"source": "文档分割策略", "category": "技术方法"}
        },
        {
            "text": "LangChain 0.3版本进行了重大重构，将核心功能拆分为多个包：langchain-core、langchain-community等，使架构更加模块化和可维护。",
            "metadata": {"source": "LangChain 0.3更新", "category": "版本信息"}
        }
    ]
    
    return documents

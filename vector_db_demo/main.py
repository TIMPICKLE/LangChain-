"""
主程序入口
提供向量数据库和检索器的示例用法
"""
import os
import argparse
from typing import List, Dict, Any

from custom_llm import CustomLLM
from local_embeddings import LocalEmbeddings
from vector_store import VectorStore
from rag_system import RAGSystem
from sample_data import load_sample_data

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="向量数据库和检索器示例")
    
    # 子命令
    subparsers = parser.add_subparsers(dest="command", help="选择要执行的操作")
    
    # 添加文档
    add_parser = subparsers.add_parser("add", help="添加文档到向量数据库")
    add_parser.add_argument("--sample", action="store_true", help="加载示例数据")
    add_parser.add_argument("--text", type=str, help="要添加的单个文本")
    add_parser.add_argument("--source", type=str, default="用户输入", help="文本来源")
    
    # 搜索文档
    search_parser = subparsers.add_parser("search", help="在向量数据库中搜索")
    search_parser.add_argument("query", type=str, help="搜索查询")
    search_parser.add_argument("--k", type=int, default=3, help="返回结果数量")
    
    # RAG查询
    query_parser = subparsers.add_parser("query", help="使用RAG系统进行查询")
    query_parser.add_argument("question", type=str, help="问题")
    
    # 清空数据库
    subparsers.add_parser("clear", help="清空向量数据库")
    
    # 测试LLM
    subparsers.add_parser("test-llm", help="测试语言模型连接")
    
    return parser.parse_args()

def test_llm():
    """测试语言模型连接"""
    llm = CustomLLM()
    print("正在测试语言模型连接...")
    response = llm.invoke("你好，请简短自我介绍")
    print(f"模型响应: {response.content}")

def main():
    """主程序入口"""
    args = parse_arguments()
    
    # 创建嵌入模型
    embedding_model = LocalEmbeddings()
    
    # 创建向量存储
    vector_store = VectorStore(embedding_model=embedding_model)
    
    # 创建RAG系统
    rag = RAGSystem(vector_store=vector_store)
    
    if args.command == "add":
        # 添加文档
        if args.sample:
            # 加载示例数据
            print("正在加载示例数据...")
            documents = load_sample_data()
            texts = [doc["text"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]
            
            # 添加到向量存储
            rag.add_documents(texts, metadatas)
            print(f"已添加 {len(texts)} 个示例文档到向量数据库")
        
        elif args.text:
            # 添加单个文本
            rag.add_documents(
                [args.text], 
                [{"source": args.source}]
            )
            print("已添加文本到向量数据库")
    
    elif args.command == "search":
        # 搜索文档
        print(f"正在搜索: {args.query}")
        results = rag.search(args.query, k=args.k)
        
        print(f"找到 {len(results)} 个相关文档:")
        for i, doc in enumerate(results):
            print(f"\n--- 结果 {i+1} ---")
            print(f"内容: {doc.page_content}")
            print(f"元数据: {doc.metadata}")
    
    elif args.command == "query":
        # RAG查询
        print(f"问题: {args.question}")
        answer = rag.query(args.question)
        print("\n回答:")
        print(answer)
    
    elif args.command == "clear":
        # 清空数据库
        vector_store.delete_collection()
        print("已清空向量数据库")
    
    elif args.command == "test-llm":
        # 测试LLM
        test_llm()
    
    else:
        print("请指定要执行的操作。使用 --help 查看帮助。")

if __name__ == "__main__":
    main()

"""
高级文档加载示例
演示如何加载不同类型的文档并添加到向量数据库
"""
import os
import argparse
from typing import List, Optional

from document_processor import DocumentProcessor
from vector_store import VectorStore
from local_embeddings import LocalEmbeddings

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="加载文档到向量数据库")
    
    # 子命令
    subparsers = parser.add_subparsers(dest="command", help="选择要执行的操作")
    
    # 加载文本文件
    text_parser = subparsers.add_parser("load-text", help="加载文本文件")
    text_parser.add_argument("file_path", type=str, help="文本文件路径")
    
    # 加载PDF文件
    pdf_parser = subparsers.add_parser("load-pdf", help="加载PDF文件")
    pdf_parser.add_argument("file_path", type=str, help="PDF文件路径")
    
    # 加载目录
    dir_parser = subparsers.add_parser("load-dir", help="加载目录")
    dir_parser.add_argument("directory", type=str, help="目录路径")
    dir_parser.add_argument("--pattern", type=str, default="**/*.txt", help="文件匹配模式")
    
    # 通用参数
    for p in [text_parser, pdf_parser, dir_parser]:
        p.add_argument("--chunk-size", type=int, default=1000, help="分块大小")
        p.add_argument("--chunk-overlap", type=int, default=200, help="分块重叠大小")
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_arguments()
    
    if not args.command:
        print("请指定要执行的操作。使用 --help 查看帮助。")
        return
    
    # 创建文档处理器
    processor = DocumentProcessor(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    # 创建嵌入模型和向量存储
    embedding_model = LocalEmbeddings()
    vector_store = VectorStore(embedding_model=embedding_model)
    
    documents = []
    
    if args.command == "load-text":
        # 加载文本文件
        print(f"正在加载文本文件: {args.file_path}")
        documents = processor.load_text_file(args.file_path)
    
    elif args.command == "load-pdf":
        # 加载PDF文件
        print(f"正在加载PDF文件: {args.file_path}")
        documents = processor.load_pdf_file(args.file_path)
    
    elif args.command == "load-dir":
        # 加载目录
        print(f"正在加载目录: {args.directory}")
        documents = processor.load_directory(args.directory, args.pattern)
    
    # 添加到向量存储
    if documents:
        print(f"处理了 {len(documents)} 个文档片段")
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        vector_store.add_texts(texts, metadatas)
        print("已成功添加到向量数据库")
    else:
        print("没有找到任何文档")

if __name__ == "__main__":
    main()

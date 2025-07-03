"""
文档处理工具
用于加载、分割和处理文档
"""
import os
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    PyPDFLoader
)

class DocumentProcessor:
    """文档处理器，用于加载和分割文档"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """初始化文档处理器"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # 创建文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def load_text_file(self, file_path: str) -> List[Document]:
        """加载单个文本文件"""
        loader = TextLoader(file_path)
        documents = loader.load()
        return self.split_documents(documents)
    
    def load_pdf_file(self, file_path: str) -> List[Document]:
        """加载单个PDF文件"""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return self.split_documents(documents)
    
    def load_directory(
        self, 
        directory_path: str, 
        glob_pattern: str = "**/*.txt"
    ) -> List[Document]:
        """加载目录中的所有文本文件"""
        loader = DirectoryLoader(
            directory_path,
            glob=glob_pattern,
            loader_cls=TextLoader
        )
        documents = loader.load()
        return self.split_documents(documents)
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档"""
        return self.text_splitter.split_documents(documents)
    
    def split_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """分割单个文本"""
        if metadata is None:
            metadata = {}
        return self.text_splitter.create_documents([text], [metadata])
    
    def process_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """处理单个文本"""
        return self.split_text(text, metadata)

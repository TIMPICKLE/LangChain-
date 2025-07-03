"""
向量数据库和检索器
使用FAISS作为向量存储引擎
"""
import os
import pickle
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from local_embeddings import LocalEmbeddings

class VectorStore:
    """向量存储和检索类"""
    
    def __init__(
        self, 
        embedding_model=None, 
        persist_directory="vector_db",
        collection_name="default_collection"
    ):
        """初始化向量存储"""
        # 确保使用绝对路径
        if not os.path.isabs(persist_directory):
            persist_directory = os.path.join(os.getcwd(), persist_directory)
        
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.collection_path = os.path.join(persist_directory, collection_name)
        
        # 如果没有提供嵌入模型，则使用默认本地模型
        self.embedding_model = embedding_model or LocalEmbeddings()
        
        # 确保存储目录和集合目录都存在
        os.makedirs(self.persist_directory, exist_ok=True)
        os.makedirs(self.collection_path, exist_ok=True)
        
        # 加载或创建向量存储
        self.vector_store = self._load_or_create_vector_store()
    
    def _load_or_create_vector_store(self):
        """加载或创建新的向量存储"""
        # FAISS.save_local会保存 index.faiss 和 index.pkl 两个文件
        index_file = os.path.join(self.collection_path, "index.faiss")
        if os.path.exists(index_file):
            try:
                print(f"正在加载现有向量存储: {self.collection_path}")
                # 允许反序列化
                return FAISS.load_local(
                    self.collection_path, 
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"加载向量存储失败: {e}")
                print("将创建新的向量存储")
        
        # 创建空的向量存储
        return FAISS.from_documents(
            documents=[Document(page_content="初始化文档", metadata={})],
            embedding=self.embedding_model
        )
    
    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] = None) -> List[str]:
        """添加文本到向量存储"""
        if not texts:
            return []
        
        # 如果没有提供元数据，创建空的元数据
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # 创建Document对象
        documents = [
            Document(page_content=text, metadata=metadata)
            for text, metadata in zip(texts, metadatas)
        ]
        
        # 添加到向量存储
        ids = self.vector_store.add_documents(documents)
        
        # 保存向量存储
        self._save_vector_store()
        
        return ids
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """执行相似度搜索"""
        return self.vector_store.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """执行带评分的相似度搜索"""
        return self.vector_store.similarity_search_with_score(query, k=k)
    

    def _save_vector_store(self):
        """将向量存储保存到本地"""
        if self.vector_store:
            # 确保目录存在，即使已经在 __init__ 中创建过
            # 有时候可能在程序运行过程中目录被删除
            try:
                if not os.path.exists(self.collection_path):
                    print(f"创建向量存储目录: {self.collection_path}")
                    os.makedirs(self.collection_path, exist_ok=True)
                
                self.vector_store.save_local(self.collection_path)
                print(f"向量存储已保存到 {self.collection_path}")
            except Exception as e:
                print(f"保存向量存储时出错: {str(e)}")
                # 提供更多诊断信息
                print(f"目录路径: {self.collection_path}")
                print(f"目录是否存在: {os.path.exists(os.path.dirname(self.collection_path))}")
                print(f"是否可写: {os.access(os.path.dirname(self.collection_path), os.W_OK) if os.path.exists(os.path.dirname(self.collection_path)) else False}")
                raise
    
    def delete_collection(self):
        """删除整个集合"""
        import shutil
        if os.path.exists(self.collection_path):
            shutil.rmtree(self.collection_path)
            print(f"已删除集合: {self.collection_name}")
            # 重新创建一个空的向量存储
            self.vector_store = FAISS.from_documents(
                documents=[Document(page_content="初始化文档", metadata={})],
                embedding=self.embedding_model
            )
        else:
            print(f"集合不存在: {self.collection_name}")

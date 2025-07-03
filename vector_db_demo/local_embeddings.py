"""
本地文本嵌入模型
使用sentence-transformers本地生成文本嵌入
"""
from typing import List
import numpy as np
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

class LocalEmbeddings(Embeddings):
    """本地文本嵌入模型，使用sentence_transformers"""
    
    def __init__(
        self, 
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        cache_folder: str = None
    ):
        """初始化模型"""
        self.model_name = model_name
        self.model = None
        
        # 检查本地Models文件夹中是否有模型
        import os
        local_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Models")
        model_loaded = False
        
        if os.path.exists(local_model_path):
            model_dir_name = f"models--{model_name.replace('/', '--')}"
            local_model_full_path = os.path.join(local_model_path, model_dir_name)
            # 查找模型实际存储的snapshot文件夹
            snapshots_dir = os.path.join(local_model_full_path, "snapshots")
            if os.path.exists(snapshots_dir) and os.listdir(snapshots_dir):
                # 获取第一个snapshot目录
                snapshot_id = os.listdir(snapshots_dir)[0]
                local_snapshot_path = os.path.join(snapshots_dir, snapshot_id)
                if os.path.exists(local_snapshot_path):
                    print(f"使用本地模型: {local_snapshot_path}")
                    try:
                        # 直接使用本地模型路径而不是原始模型名称
                        self.model = SentenceTransformer(
                            local_snapshot_path, 
                            cache_folder=local_model_path,
                            local_files_only=True  # 强制使用本地文件
                        )
                        model_loaded = True
                    except Exception as e:
                        print(f"加载本地模型失败: {str(e)}")
        
        # 如果本地模型没有加载成功，尝试在线加载
        if not model_loaded:
            print(f"尝试从huggingface加载模型: {model_name}")
            try:
                self.model = SentenceTransformer(model_name, cache_folder=cache_folder)
            except Exception as e:
                print(f"无法从huggingface加载模型: {str(e)}")
                raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """获取文档列表的嵌入向量"""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """获取查询的嵌入向量"""
        embedding = self.model.encode(text)
        return embedding.tolist()

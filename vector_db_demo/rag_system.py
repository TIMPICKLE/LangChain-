"""
检索增强生成（RAG）系统
结合向量数据库和语言模型
"""
from typing import List, Dict, Any
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough

from custom_llm import CustomLLM
from vector_store import VectorStore

class RAGSystem:
    """检索增强生成系统"""
    
    def __init__(
        self, 
        vector_store=None,
        llm=None
    ):
        """初始化RAG系统"""
        # 初始化向量存储和语言模型
        self.vector_store = vector_store or VectorStore()
        self.llm = llm or CustomLLM()
        
        # 创建RAG链
        self.chain = self._create_rag_chain()
    
    def _create_rag_chain(self):
        """创建RAG检索链"""
        # 定义检索器
        retriever = self.vector_store.vector_store.as_retriever()
        
        # 定义提示模板
        template = """你是一个有用的AI助手。使用以下上下文片段回答用户的问题。
        如果你不知道答案，只需说你不知道，不要试图编造答案。
        
        上下文:
        {context}
        
        问题: {question}
        
        请提供详细且有帮助的回答:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # 构建检索链
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain
    
    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]] = None) -> List[str]:
        """添加文档到向量存储"""
        return self.vector_store.add_texts(texts, metadatas)
    
    def query(self, question: str) -> str:
        """执行RAG查询"""
        return self.chain.invoke(question)
    
    def search(self, query: str, k: int = 4) -> List[Document]:
        """直接搜索相关文档，不使用语言模型"""
        return self.vector_store.similarity_search(query, k=k)

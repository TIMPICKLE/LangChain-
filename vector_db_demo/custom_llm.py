"""
自定义语言模型接口
使用内部部署的LLM服务来代替OpenAI
"""
import os
import time
import json
import requests
from typing import Any, List, Dict, Optional
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatResult, ChatGeneration

# 加载环境变量
load_dotenv()

class CustomLLM(BaseChatModel):
    """自定义的LLM类，适配内部API服务"""
    
    api_base_url: str = os.getenv("API_BASE_URL", "http://10.6.12.215:6091/v1")
    model_name: str = os.getenv("API_MODEL_NAME", "DeepSeek-V3-0324-HSW")
    temperature: float = 0.7
    max_tokens: int = 1000
    
    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_id: Optional[str] = None, **kwargs: Any
    ) -> ChatResult:
        """生成文本响应"""
        # 构建请求URL
        url = f"{self.api_base_url}/chat/completions"
        
        # 构建请求头
        headers = {
            "Content-Type": "application/json"
        }
        
        # 格式化消息
        formatted_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                formatted_messages.append({
                    "role": "user",
                    "content": message.content
                })
            elif isinstance(message, AIMessage):
                formatted_messages.append({
                    "role": "assistant",
                    "content": message.content
                })
            elif isinstance(message, SystemMessage):
                formatted_messages.append({
                    "role": "system",
                    "content": message.content
                })
        
        # 构建请求体
        data = {
            "model": self.model_name,
            "messages": formatted_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        if stop:
            data["stop"] = stop
        
        # 更新额外参数
        data.update({k: v for k, v in kwargs.items() if k not in ["run_manager"]})
        
        try:
            start_time = time.time()
            response = requests.post(url, headers=headers, json=data, timeout=30)
            end_time = time.time()
            
            # 如果请求成功
            if response.status_code == 200:
                response_json = response.json()
                content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                # 创建结果对象
                message = AIMessage(content=content)
                generation = ChatGeneration(message=message)
                return ChatResult(generations=[generation])
            else:
                raise ValueError(f"API请求失败: {response.status_code}, {response.text}")
        except requests.exceptions.RequestException as e:
            raise ValueError(f"请求发生错误: {e}")
    
    @property
    def _llm_type(self) -> str:
        """返回LLM类型"""
        return "custom_llm"

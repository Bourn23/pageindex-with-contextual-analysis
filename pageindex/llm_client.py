"""
LLM Client abstraction layer supporting multiple providers (OpenAI, Gemini, etc.)
"""
import os
import logging
import time
import asyncio
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """Unified LLM client supporting multiple providers"""
    
    def __init__(self, provider=None, api_key=None, model=None):
        """
        Initialize LLM client
        
        Args:
            provider: 'openai' or 'gemini'. If None, auto-detect from env vars
            api_key: API key for the provider. If None, read from env
            model: Default model to use
        """
        self.provider = provider or self._detect_provider()
        self.api_key = api_key or self._get_api_key()
        self.default_model = model
        
        # Initialize client based on provider
        if self.provider == 'gemini':
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
            self.async_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
        else:  # openai
            self.client = OpenAI(api_key=self.api_key)
            self.async_client = AsyncOpenAI(api_key=self.api_key)
    
    def _detect_provider(self):
        """Auto-detect provider from environment variables"""
        if os.getenv("GEMINI_API_KEY"):
            return 'gemini'
        elif os.getenv("CHATGPT_API_KEY") or os.getenv("OPENAI_API_KEY"):
            return 'openai'
        else:
            # Default to OpenAI
            return 'openai'
    
    def _get_api_key(self):
        """Get API key based on provider"""
        if self.provider == 'gemini':
            key = os.getenv("GEMINI_API_KEY")
            if not key:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            return key
        else:
            key = os.getenv("CHATGPT_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError("CHATGPT_API_KEY or OPENAI_API_KEY not found in environment variables")
            return key
    
    def chat_completion(self, model, prompt, chat_history=None, temperature=0, max_retries=10):
        """
        Synchronous chat completion
        
        Args:
            model: Model name to use
            prompt: User prompt
            chat_history: Optional list of previous messages
            temperature: Sampling temperature
            max_retries: Maximum number of retry attempts
            
        Returns:
            Response content string
        """
        for i in range(max_retries):
            try:
                if chat_history:
                    messages = chat_history.copy()
                    messages.append({"role": "user", "content": prompt})
                else:
                    messages = [{"role": "user", "content": prompt}]
                
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
                
                return response.choices[0].message.content
            except Exception as e:
                print('************* Retrying *************')
                logging.error(f"Error: {e}")
                if i < max_retries - 1:
                    time.sleep(1)
                else:
                    logging.error(f'Max retries reached for prompt: {prompt[:100]}...')
                    return "Error"
    
    def chat_completion_with_finish_reason(self, model, prompt, chat_history=None, temperature=0, max_retries=10):
        """
        Synchronous chat completion with finish reason
        
        Args:
            model: Model name to use
            prompt: User prompt
            chat_history: Optional list of previous messages
            temperature: Sampling temperature
            max_retries: Maximum number of retry attempts
            
        Returns:
            Tuple of (response_content, finish_reason)
        """
        for i in range(max_retries):
            try:
                if chat_history:
                    messages = chat_history.copy()
                    messages.append({"role": "user", "content": prompt})
                else:
                    messages = [{"role": "user", "content": prompt}]
                
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
                
                finish_reason = response.choices[0].finish_reason
                content = response.choices[0].message.content
                
                if finish_reason == "length":
                    return content, "max_output_reached"
                else:
                    return content, "finished"
                    
            except Exception as e:
                print('************* Retrying *************')
                logging.error(f"Error: {e}")
                if i < max_retries - 1:
                    time.sleep(1)
                else:
                    logging.error(f'Max retries reached for prompt: {prompt[:100]}...')
                    return "Error", "error"
    
    async def chat_completion_async(self, model, prompt, temperature=0, max_retries=10):
        """
        Asynchronous chat completion
        
        Args:
            model: Model name to use
            prompt: User prompt
            temperature: Sampling temperature
            max_retries: Maximum number of retry attempts
            
        Returns:
            Response content string
        """
        messages = [{"role": "user", "content": prompt}]
        
        for i in range(max_retries):
            try:
                response = await self.async_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
                return response.choices[0].message.content
            except Exception as e:
                print('************* Retrying *************')
                logging.error(f"Error: {e}")
                if i < max_retries - 1:
                    await asyncio.sleep(1)
                else:
                    logging.error(f'Max retries reached for prompt: {prompt[:100]}...')
                    return "Error"


# Global client instance
_global_client = None


def get_llm_client(provider=None, api_key=None, model=None):
    """
    Get or create global LLM client instance
    
    Args:
        provider: 'openai' or 'gemini'. If None, auto-detect
        api_key: API key. If None, read from env
        model: Default model
        
    Returns:
        LLMClient instance
    """
    global _global_client
    if _global_client is None:
        _global_client = LLMClient(provider=provider, api_key=api_key, model=model)
    return _global_client


def set_llm_provider(provider, api_key=None):
    """
    Set the LLM provider globally
    
    Args:
        provider: 'openai' or 'gemini'
        api_key: Optional API key (will use env var if not provided)
    """
    global _global_client
    _global_client = LLMClient(provider=provider, api_key=api_key)

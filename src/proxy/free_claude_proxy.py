"""
Free Claude Proxy for Atlas v4.0
Connects to NVIDIA NIM API for Claude access
"""

import os
import aiohttp
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Message:
    """Chat message"""
    role: str
    content: str


class FreeClaudeProxy:
    """
    Free Claude proxy using NVIDIA NIM API.
    
    Provides access to Claude through NVIDIA's free tier.
    Falls back to rule-based analysis if unavailable.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY") or os.getenv("NIM_API_KEY")
        self.base_url = "https://integrate.api.nvidia.com/v1"
        self.model = "anthropic/claude-3-haiku"
        
        self._available = bool(self.api_key)
        
        if self._available:
            print("✓ NVIDIA NIM API connected")
        else:
            print("⚠ No API key found, using rule-based analysis")
    
    @property
    def messages(self):
        """Return messages interface"""
        return MessagesInterface(self)
    
    def is_available(self) -> bool:
        """Check if LLM is available"""
        return self._available
    
    async def create_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """Create a completion"""
        
        if not self._available:
            return {
                "content": [{"text": "LLM not available, using rule-based analysis"}],
                "model": "fallback"
            }
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
                
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        
                        # Convert OpenAI format to our format
                        return {
                            "content": [{"text": data["choices"][0]["message"]["content"]}],
                            "model": data.get("model", self.model)
                        }
                    else:
                        error_text = await resp.text()
                        print(f"API error: {resp.status} - {error_text}")
                        return {
                            "content": [{"text": "API error occurred"}],
                            "model": "error"
                        }
        
        except Exception as e:
            print(f"API exception: {e}")
            return {
                "content": [{"text": f"Error: {str(e)}"}],
                "model": "error"
            }


class MessagesInterface:
    """Messages interface for Claude-style API"""
    
    def __init__(self, client: FreeClaudeProxy):
        self.client = client
    
    async def create(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """Create messages completion"""
        return await self.client.create_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )

"""
Ollama Client Utility
Provides Llama 3 integration as fallback/alternative to Gemini API
"""

import requests
import json
from typing import Optional, Dict, Any


class OllamaClient:
    """Client for interacting with Ollama local LLM server."""

    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama client.

        Args:
            model: Model name (default: llama3)
            base_url: Ollama server URL
        """
        self.model = model
        self.base_url = base_url
        self.generate_url = f"{base_url}/api/generate"
        self.chat_url = f"{base_url}/api/chat"

    def generate(self, prompt: str, system: Optional[str] = None,
                 temperature: float = 0.7, max_tokens: int = 500) -> Optional[str]:
        """
        Generate text completion using Ollama.

        Args:
            prompt: Input prompt
            system: Optional system message
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text or None if failed
        """
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }

            if system:
                payload["system"] = system

            response = requests.post(
                self.generate_url,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                print(f"[Ollama] Error {response.status_code}: {response.text}")
                return None

        except requests.exceptions.ConnectionError:
            print("[Ollama] Connection error. Is Ollama running? Start with: ollama serve")
            return None
        except Exception as e:
            print(f"[Ollama] Generation error: {e}")
            return None

    def chat(self, messages: list, temperature: float = 0.7,
             max_tokens: int = 500) -> Optional[str]:
        """
        Chat completion using Ollama.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated response or None if failed
        """
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }

            response = requests.post(
                self.chat_url,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("message", {}).get("content", "").strip()
            else:
                print(f"[Ollama] Error {response.status_code}: {response.text}")
                return None

        except requests.exceptions.ConnectionError:
            print("[Ollama] Connection error. Is Ollama running? Start with: ollama serve")
            return None
        except Exception as e:
            print(f"[Ollama] Chat error: {e}")
            return None

    def is_available(self) -> bool:
        """
        Check if Ollama server is available.

        Returns:
            True if server is reachable, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False


# Global instance
ollama_client = OllamaClient(model="llama3")


def generate_with_fallback(prompt: str, gemini_func, system: Optional[str] = None) -> str:
    """
    Try Gemini first, fall back to Ollama if it fails.

    Args:
        prompt: Input prompt
        gemini_func: Function that calls Gemini API
        system: Optional system message for Ollama

    Returns:
        Generated text from either Gemini or Ollama
    """
    import time

    # Try Gemini with retry logic
    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            result = gemini_func()
            if result:
                return result
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                if attempt < max_retries - 1:
                    print(f"[Hybrid LLM] Gemini quota hit, retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    print("[Hybrid LLM] Gemini quota exhausted, switching to Ollama (Llama 3)...")
                    break
            elif "API_KEY" in str(e) or "ADC" in str(e):
                print("[Hybrid LLM] Gemini API key not configured, using Ollama (Llama 3)...")
                break
            else:
                print(f"[Hybrid LLM] Gemini error: {e}, trying Ollama...")
                break

    # Fallback to Ollama
    if ollama_client.is_available():
        print("[Hybrid LLM] Using Ollama (Llama 3) as LLM backend...")
        result = ollama_client.generate(prompt, system=system, temperature=0.7, max_tokens=500)
        if result:
            return result

    # If both fail, return empty string
    print("[Hybrid LLM] Both Gemini and Ollama failed, using empty response")
    return ""

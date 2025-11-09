"""
Cloud backend router for TagOmatic

Supports multiple LLM providers:
- Remote Ollama (external Ollama instance)
- OpenAI-compatible (LLMStudio, vLLM, etc.)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from urllib import request, parse, error
import json


class BaseProvider(ABC):
    """Base class for all LLM providers"""
    
    def __init__(self, name: str):
        self.name = name
        self._configured = False
    
    @abstractmethod
    def is_configured(self) -> bool:
        """Check if provider has required configuration"""
        pass
    
    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, Any]],
        images_b64: Optional[List[str]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send chat request. Returns dict with 'message' key containing content."""
        pass
    
    def _post_json(
        self,
        url: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, str]] = None,
        payload: Optional[Dict[str, Any]] = None,
        timeout: int = 60,
    ) -> Dict[str, Any]:
        """Helper for HTTP POST JSON requests"""
        if params:
            qs = parse.urlencode(params)
            joiner = '&' if ('?' in url) else '?'
            url = f"{url}{joiner}{qs}"
        
        data = json.dumps(payload or {}).encode('utf-8')
        req = request.Request(url, data=data, method='POST')
        
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        req.add_header('Content-Type', 'application/json')
        
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
                text = body.decode('utf-8', errors='replace')
            return json.loads(text)
        except error.HTTPError as e:
            try:
                body = e.read().decode('utf-8', errors='replace')
            except Exception:
                body = ''
            raise RuntimeError(f"HTTP {e.code} {e.reason} -> {body[:400]}")
    
    def _extract_images_from_messages(
        self,
        messages: List[Dict[str, Any]],
        images_b64: Optional[List[str]] = None,
    ) -> List[str]:
        """Extract base64 images from messages or explicit list"""
        if images_b64 is not None:
            return [b for b in images_b64 if isinstance(b, str) and b]
        
        images = []
        for m in messages or []:
            imgs = m.get("images") if isinstance(m, dict) else None
            if isinstance(imgs, list):
                for b in imgs:
                    if isinstance(b, str) and b:
                        images.append(b)
        return images


class RemoteOllamaProvider(BaseProvider):
    """Remote Ollama instance provider"""
    
    def __init__(self):
        super().__init__("Remote Ollama")
        self.host: str = "http://localhost:11434"
        self.model: str = ""
    
    def configure(
        self,
        *,
        host: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        if host is not None:
            self.host = str(host or "http://localhost:11434").rstrip('/')
        if model is not None:
            self.model = str(model or "")
    
    def is_configured(self) -> bool:
        return bool(self.host and self.model)
    
    def chat(
        self,
        messages: List[Dict[str, Any]],
        images_b64: Optional[List[str]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.is_configured():
            raise RuntimeError("Remote Ollama not configured: missing host or model")
        
        url = f"{self.host}/api/chat"
        images = self._extract_images_from_messages(messages, images_b64)
        
        payload_msgs = []
        for m in messages or []:
            msg_data = {
                'role': m.get('role', 'user'),
                'content': str(m.get('content', '')),
                'images': m.get('images') or (images if not payload_msgs else [])
            }
            payload_msgs.append(msg_data)
            if payload_msgs:
                images = []  # Don't re-attach
        
        opts = {
            'temperature': float(options.get('temperature', 0.2)) if options else 0.2,
            'top_p': float(options.get('top_p', 0.8)) if options else 0.8,
            'num_predict': int(options.get('num_predict', 192)) if options else 192,
        }
        
        body = {
            'model': self.model,
            'messages': payload_msgs,
            'stream': False,
            'options': opts,
        }
        
        data = self._post_json(url, payload=body, timeout=60)
        return {
            "message": data.get('message') or {"role": "assistant", "content": ""},
            "_provider": f"Ollama: {self.model} @ {self.host}"
        }


class OpenAICompatibleProvider(BaseProvider):
    """OpenAI-compatible API provider (LLMStudio, vLLM, etc.)"""
    
    def __init__(self):
        super().__init__("OpenAI-Compatible")
        self.base_url: str = "http://localhost:1234/v1"
        self.api_key: str = ""
        self.model: str = ""
    
    def configure(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        if base_url is not None:
            self.base_url = str(base_url or "http://localhost:1234/v1").rstrip('/')
        if api_key is not None:
            self.api_key = str(api_key or "")
        if model is not None:
            self.model = str(model or "")
        
        # Fallback: try to read model from QSettings if not set
        if not self.model:
            try:
                from PySide6.QtCore import QSettings
                s = QSettings('Pistonspy', 'TagOmatic')
                self.model = s.value('cloud/oai_compat_model', '', type=str) or ''
            except Exception:
                pass
    
    def is_configured(self) -> bool:
        return bool(self.base_url and self.model)
    
    def chat(
        self,
        messages: List[Dict[str, Any]],
        images_b64: Optional[List[str]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.model:
            raise ValueError(
                'OpenAI-compatible model not set. '
                'Choose a model in Cloud Backends and Save.'
            )
        
        url = f"{self.base_url}/chat/completions"
        images = self._extract_images_from_messages(messages, images_b64)
        
        oai_msgs = []
        for m in messages or []:
            content_parts = []
            if m.get('content'):
                content_parts.append({"type": "text", "text": str(m.get('content'))})
            
            # Add images from message
            for b64 in (m.get('images') or []):
                if isinstance(b64, str) and len(b64) > 10:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                    })
            
            # Attach remaining images to first message
            if images and not oai_msgs:
                for b64 in images:
                    if isinstance(b64, str) and len(b64) > 10:
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                        })
            
            oai_msgs.append({
                "role": m.get('role', 'user'),
                "content": content_parts
            })
        
        opts = options or {}
        body = {
            "model": self.model,
            "messages": oai_msgs,
            "stream": False,
            "temperature": float(opts.get('temperature', 0.2)),
            "top_p": float(opts.get('top_p', 0.8)),
            "max_tokens": int(opts.get('num_predict', 192)),
        }
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        data = self._post_json(url, headers=headers, payload=body, timeout=60)
        
        text = ''
        try:
            text = str(data["choices"][0]["message"]["content"])
        except Exception:
            text = json.dumps(data)
        
        provider_name = "LLMStudio" if "localhost:1234" in self.base_url else "OpenAI-Compatible"
        return {
            "message": {"content": text},
            "_provider": f"{provider_name}: {self.model}"
        }


class CloudRouter:
    """Router for cloud LLM providers"""
    
    def __init__(self):
        self.use_cloud: bool = False
        self.preferred_provider: str = "auto"
        
        # Initialize providers (no OpenAI)
        self.remote_ollama = RemoteOllamaProvider()
        self.openai_compat = OpenAICompatibleProvider()
    
    def configure(
        self,
        *,
        use_cloud: Optional[bool] = None,
        preferred_provider: Optional[str] = None,
        # Remote Ollama
        external_ollama_host: Optional[str] = None,
        external_ollama_model: Optional[str] = None,
        # OpenAI-compatible
        oai_compat_base_url: Optional[str] = None,
        oai_compat_api_key: Optional[str] = None,
        oai_compat_model: Optional[str] = None,
    ) -> None:
        """Configure router and providers"""
        if use_cloud is not None:
            self.use_cloud = bool(use_cloud)
        
        if preferred_provider is not None:
            val = str(preferred_provider or 'auto').lower()
            valid = {"auto", "remote-ollama", "openai-compatible", "llmstudio"}
            self.preferred_provider = val if val in valid else "auto"
        
        # Configure Remote Ollama
        self.remote_ollama.configure(
            host=external_ollama_host,
            model=external_ollama_model,
        )
        
        # Configure OpenAI-compatible
        self.openai_compat.configure(
            base_url=oai_compat_base_url,
            api_key=oai_compat_api_key,
            model=oai_compat_model,
        )
    
    def _select_provider(self) -> Optional[BaseProvider]:
        """Select the appropriate provider based on configuration"""
        if not self.use_cloud:
            return None
        
        # Try preferred provider first
        if self.preferred_provider == "remote-ollama" and self.remote_ollama.is_configured():
            return self.remote_ollama
        elif self.preferred_provider in ("openai-compatible", "llmstudio") and self.openai_compat.is_configured():
            return self.openai_compat
        
        # Auto-detect: try in order of preference
        if self.openai_compat.is_configured():
            return self.openai_compat
        if self.remote_ollama.is_configured():
            return self.remote_ollama
        
        return None
    
    def chat(
        self,
        messages: List[Dict[str, Any]],
        images_b64: Optional[List[str]] = None,
        options_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Dispatch chat request to configured provider"""
        provider = self._select_provider()
        if not provider:
            raise RuntimeError("Cloud not configured or no provider available")
        
        return provider.chat(messages, images_b64, options_override)


# Singleton instance
cloud_router = CloudRouter()

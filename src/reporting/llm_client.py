"""
LLM 客户端 - 抽象多平台 LLM API 调用

支持的提供商（均兼容 OpenAI SDK 格式）：
  - openai:    OpenAI 官方
  - deepseek:  DeepSeek（国内直连，性价比高）
  - zhipu:     智谱 GLM（国内直连）
  - moonshot:  Moonshot / Kimi（国内直连，长上下文）
  - claude:    Anthropic Claude
  - local:     本地模型（Ollama）

所有 API Key 通过环境变量或 .env 文件读取，settings.yaml 中不存放密钥。
"""
import os
import time
from pathlib import Path


def _load_dotenv():
    """加载 .env 文件到环境变量（如果存在）"""
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # 去除引号包裹
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            if key and value and key not in os.environ:
                os.environ[key] = value


# 模块加载时自动读取 .env
_load_dotenv()

# 提供商 → 环境变量名 映射
PROVIDER_ENV_MAP = {
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "zhipu": "ZHIPU_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
}

# 网络相关异常类型（用于区分可重试的网络错误）
_RETRIABLE_ERRORS = (
    ConnectionError,
    TimeoutError,
    OSError,
)


def _is_retriable_error(e: Exception) -> bool:
    """判断异常是否为可重试的临时性错误（网络/超时/服务端）"""
    # Python 内置网络错误
    if isinstance(e, _RETRIABLE_ERRORS):
        return True

    # OpenAI SDK 特有错误
    try:
        import openai
        if isinstance(e, (openai.APITimeoutError, openai.APIConnectionError)):
            return True
        # 服务端 5xx 错误，可重试
        if isinstance(e, openai.InternalServerError):
            return True
        # 限流，可重试
        if isinstance(e, openai.RateLimitError):
            return True
    except ImportError:
        pass

    # Anthropic SDK 特有错误
    try:
        import anthropic
        if isinstance(e, (anthropic.APITimeoutError, anthropic.APIConnectionError)):
            return True
        if isinstance(e, anthropic.InternalServerError):
            return True
        if isinstance(e, anthropic.RateLimitError):
            return True
    except ImportError:
        pass

    return False


class LLMClient:
    """LLM API 客户端（统一接口）

    所有兼容 OpenAI SDK 的提供商（DeepSeek / 智谱 / Moonshot）
    共用同一套 OpenAI SDK 调用逻辑，仅 base_url 和 model 不同。
    Claude 使用独立的 Anthropic SDK。本地模型使用 Ollama REST API。
    """

    # 最大重试次数
    MAX_RETRIES = 3
    # 默认请求超时（秒）：连接超时, 读取超时
    DEFAULT_TIMEOUT = (15, 300)

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.provider = self.config.get("provider", "openai")
        self._openai_client = None
        self._claude_client = None
        # 从配置读取超时，未配置则使用默认值
        timeout_cfg = self.config.get("timeout")
        if isinstance(timeout_cfg, (list, tuple)) and len(timeout_cfg) == 2:
            self.TIMEOUT = tuple(timeout_cfg)
        elif isinstance(timeout_cfg, (int, float)):
            self.TIMEOUT = (timeout_cfg, timeout_cfg)
        else:
            self.TIMEOUT = self.DEFAULT_TIMEOUT

    # ----------------------------------------------------------
    # 统一入口
    # ----------------------------------------------------------
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """调用 LLM 生成文本（含重试与空响应防护）

        重试策略：
        - 网络错误（ConnectionError / Timeout / OSError）：指数退避重试
        - 空响应：重试
        - 其他错误（认证/参数等）：不重试，直接抛出
        """
        last_error = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                if self.provider == "claude":
                    content = self._call_claude(system_prompt, user_prompt)
                elif self.provider == "local":
                    content = self._call_local(system_prompt, user_prompt)
                elif self.provider in PROVIDER_ENV_MAP or self.provider == "openai":
                    content = self._call_openai_compatible(system_prompt, user_prompt)
                else:
                    raise ValueError(
                        f"不支持的 LLM 提供商: {self.provider}\n"
                        f"可选: openai / deepseek / zhipu / moonshot / claude / local"
                    )

                # 防护：检查空响应
                if not content or not content.strip():
                    print(f"[LLM 警告] 第 {attempt} 次调用返回空内容，"
                          f"provider={self.provider}")
                    last_error = "LLM 返回空内容"
                    if attempt < self.MAX_RETRIES:
                        time.sleep(2 * attempt)  # 指数退避
                        continue
                    else:
                        raise RuntimeError(
                            f"LLM 连续 {self.MAX_RETRIES} 次返回空内容，"
                            f"请检查 API 配置和网络连接"
                        )

                return content

            except RuntimeError:
                raise  # 重试耗尽，向上抛出
            except Exception as e:
                if _is_retriable_error(e) or isinstance(e, _RETRIABLE_ERRORS):
                    # 可重试的临时性错误（网络/超时/5xx/限流）
                    wait = 3 * (2 ** (attempt - 1))  # 3s, 6s, 12s
                    print(f"[LLM 网络错误] 第 {attempt}/{self.MAX_RETRIES} 次失败: "
                          f"{type(e).__name__}: {e}，{wait}s 后重试...")
                    if attempt < self.MAX_RETRIES:
                        time.sleep(wait)
                    else:
                        raise RuntimeError(
                            f"LLM 调用连续 {self.MAX_RETRIES} 次失败: {e}"
                        ) from e
                else:
                    # 其他错误（认证/参数等）：不重试，直接抛出
                    print(f"[LLM 错误] 调用失败（不重试）: {type(e).__name__}: {e}")
                    raise RuntimeError(
                        f"LLM 调用失败: {e}"
                    ) from e

    # ----------------------------------------------------------
    # OpenAI 兼容接口（覆盖 OpenAI / DeepSeek / 智谱 / Moonshot）
    # ----------------------------------------------------------
    def _get_openai_compatible_client(self):
        """创建 OpenAI 兼容客户端"""
        if self._openai_client is not None:
            return self._openai_client

        from openai import OpenAI

        provider_config = self.config.get(self.provider, {})
        env_key = PROVIDER_ENV_MAP.get(self.provider, "OPENAI_API_KEY")
        api_key = os.environ.get(env_key, "")

        base_url = provider_config.get("base_url")
        if not base_url:
            # 兜底：从默认配置中取
            base_url = self.config.get("openai", {}).get("base_url", "https://api.openai.com/v1")

        self._openai_client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=self.TIMEOUT,
        )
        return self._openai_client

    def _call_openai_compatible(self, system_prompt: str, user_prompt: str) -> str:
        """调用 OpenAI 兼容 API（OpenAI / DeepSeek / 智谱 / Moonshot）"""
        client = self._get_openai_compatible_client()
        provider_config = self.config.get(self.provider, {})

        response = client.chat.completions.create(
            model=provider_config.get("model", "gpt-4"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=provider_config.get("temperature", 0.7),
            max_tokens=provider_config.get("max_tokens", 8000),
        )

        if not response.choices:
            print(f"[LLM 警告] API 返回空 choices, provider={self.provider}, "
                  f"model={provider_config.get('model', 'gpt-4')}")
            return ""

        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        content = choice.message.content

        if finish_reason == "length":
            if content and content.strip():
                print(f"[LLM 警告] 输出被截断 (finish_reason=length), "
                      f"已生成 {len(content)} 字符, provider={self.provider}。"
                      f"可尝试增大 max_tokens (当前: {provider_config.get('max_tokens', 8000)})")
                return content
            else:
                print(f"[LLM 警告] finish_reason=length 但内容为空, "
                      f"provider={self.provider}, model={provider_config.get('model', 'gpt-4')}。"
                      f"可能原因: 输入过长超出上下文窗口, 或 max_tokens 不足。"
                      f"当前 max_tokens={provider_config.get('max_tokens', 8000)}")
                return ""

        if finish_reason and finish_reason != "stop":
            print(f"[LLM 警告] 非正常结束: finish_reason={finish_reason}, "
                  f"provider={self.provider}")

        return content if content else ""

    # ----------------------------------------------------------
    # Claude（独立 SDK）
    # ----------------------------------------------------------
    def _get_claude_client(self):
        if self._claude_client is not None:
            return self._claude_client
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        base_url = self.config.get("claude", {}).get("base_url")
        kwargs = {"api_key": api_key, "timeout": self.TIMEOUT}
        if base_url:
            kwargs["base_url"] = base_url
        self._claude_client = anthropic.Anthropic(**kwargs)
        return self._claude_client

    def _call_claude(self, system_prompt: str, user_prompt: str) -> str:
        client = self._get_claude_client()
        claude_config = self.config.get("claude", {})

        response = client.messages.create(
            model=claude_config.get("model", "claude-sonnet-4-20250514"),
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=claude_config.get("temperature", 0.7),
            max_tokens=claude_config.get("max_tokens", 8000),
        )

        if not response.content:
            print(f"[LLM 警告] Claude 返回空 content, "
                  f"stop_reason={getattr(response, 'stop_reason', 'unknown')}")
            return ""

        if response.stop_reason and response.stop_reason != "end_turn":
            print(f"[LLM 警告] Claude 非正常结束: stop_reason={response.stop_reason}")

        return response.content[0].text

    # ----------------------------------------------------------
    # 本地模型（Ollama REST API）
    # ----------------------------------------------------------
    def _call_local(self, system_prompt: str, user_prompt: str) -> str:
        import requests
        local_config = self.config.get("local", {})
        base_url = local_config.get("base_url", "http://localhost:11434")

        response = requests.post(
            f"{base_url}/api/chat",
            json={
                "model": local_config.get("model", "qwen2.5:14b"),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
            },
            timeout=120,
        )
        data = response.json()
        content = data.get("message", {}).get("content", "")
        if not content:
            print(f"[LLM 警告] 本地模型返回空内容, "
                  f"response keys={list(data.keys())}")
        return content or ""

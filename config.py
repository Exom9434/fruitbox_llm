import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Model Configuration
MODEL_CONFIG = {
    # 1. GPT (OpenAI)
    "gpt": {
        "api_type": "openai",
        "model_name": "gpt-4o", # 최신 GPT-4o
        "api_key": OPENAI_API_KEY,
        "base_url": None # Default
    },
    # 2. Gemini (Google)
    "gemini": {
        "api_type": "google",
        "model_name": "gemini-1.5-pro", # 최신 1.5 Pro 또는 Flash
        "api_key": GOOGLE_API_KEY
    },
    # 3. Claude (Anthropic) - NEW
    "claude": {
        "api_type": "anthropic",
        "model_name": "claude-3-5-sonnet-20241022", # 최신 Sonnet 3.5
        "api_key": ANTHROPIC_API_KEY
    },
    # 4. DeepSeek (OpenAI Compatible) - NEW
    "deepseek": {
        "api_type": "openai", # OpenAI SDK 사용
        "model_name": "deepseek-chat", # deepseek-v3 등
        "api_key": DEEPSEEK_API_KEY,
        "base_url": "https://api.deepseek.com" # DeepSeek 엔드포인트
    },
    # 5. Llama (via Ollama Local) - NEW
    "llama": {
        "api_type": "openai", # Ollama도 OpenAI SDK 호환
        "model_name": "llama3.2", # Ollama에 설치된 모델명
        "api_key": "ollama", # 아무거나 입력
        "base_url": "http://localhost:11434/v1" # 로컬 Ollama 주소
    }
}
# llm_client.py

import json
from openai import OpenAI
import google.generativeai as genai
import anthropic
from config import MODEL_CONFIG

# Custom Functions import
from llm_strategy import (
    choose_move_with_openai_compatible, # 범용 함수
    choose_move_with_prompt_claude,     # Claude 전용
    choose_move_with_prompt_gemini,     # Gemini 전용
    BaseModel, Moves
)
from typing import Optional, Any

def call_llm(
    model_family: str,
    board_state: list,
    prompt_path: str,
    history: Optional[Any] = None,
    response_model: BaseModel = Moves,
    full_prompt_override: Optional[str] = None 
):
    """
    Unified entry point for all LLM providers.
    """
    # 1. Config 가져오기
    if model_family not in MODEL_CONFIG:
        raise ValueError(f"Unknown model family: {model_family}")
    
    config = MODEL_CONFIG[model_family]
    api_type = config["api_type"]
    model_name = config["model_name"]
    
    # 2. 라우팅 로직
    
    # [Case A] OpenAI 호환 (GPT, DeepSeek, Llama/Ollama)
    if api_type == "openai":
        client = OpenAI(
            api_key=config["api_key"],
            base_url=config.get("base_url") # base_url이 있으면 DeepSeek/Ollama 등으로 연결됨
        )
        return choose_move_with_openai_compatible(
            client=client,
            model_name=model_name,
            board_state=board_state,
            prompt_path=prompt_path,
            history=history,
            response_model=response_model,
            full_prompt_override=full_prompt_override
        )

    # [Case B] Claude (Anthropic)
    elif api_type == "anthropic":
        client = anthropic.Anthropic(api_key=config["api_key"])
        return choose_move_with_prompt_claude(
            client=client,
            model_name=model_name,
            board_state=board_state,
            prompt_path=prompt_path,
            history=history,
            response_model=response_model,
            full_prompt_override=full_prompt_override
        )

    # [Case C] Gemini (Google)
    elif api_type == "google":
        # Gemini는 함수 내부에서 genai.configure를 하거나 여기서 해도 됨
        genai.configure(api_key=config["api_key"])
        # 주의: Gemini 함수 내부에서 model_name을 config에서 가져오는지 확인 필요.
        # 현재 코드 구조상 llm_strategy 내에서 하드코딩 되어 있다면 인자로 넘겨주도록 수정 권장.
        # 여기서는 기존 호환성을 위해 strategy 함수 호출
        return choose_move_with_prompt_gemini(
            board_state, 
            prompt_path, 
            chat_session=history, 
            response_model=response_model, 
            full_prompt_override=full_prompt_override
        )

    else:
        raise NotImplementedError(f"지원되지 않는 API 타입: {api_type}")
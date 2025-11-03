#llm_client.py
import json
from config import (
    OPENAI_API_KEY, 
    ANTHROPIC_API_KEY, 
    GOOGLE_API_KEY, 
    MODEL_CONFIG
)

import json
from typing import Optional, Any
from config import MODEL_CONFIG

# Custom Functions import
from llm_strategy import (
    choose_move_with_prompt_gpt,
    choose_move_with_prompt_gemini,
    # Pydantic
    BaseModel, Moves, MovesWithCoT, MovesWithPerStepReason
)


#LLM 호출용 함수
def call_llm(
    model_family: str,
    board_state: list,
    prompt_path: str,
    history: Optional[Any] = None,
    response_model: BaseModel = Moves,
    full_prompt_override: Optional[str] = None 
):
    if model_family == "gpt":
        return choose_move_with_prompt_gpt(board_state, prompt_path, history=history, response_model=response_model, full_prompt_override=full_prompt_override)

    elif model_family == "gemini":
        #  full_prompt_override를 gemini 함수로 전달
        return choose_move_with_prompt_gemini(board_state, prompt_path, chat_session=history, response_model=response_model, full_prompt_override=full_prompt_override)

    else:
        raise NotImplementedError(f"지원되지 않는 모델: [{model_family}]")

    
### Only for Single Turns
def get_moves_only(model_family, board_state, prompt_path=None):

    system_prompt, user_prompt, raw_response, moves, _ = call_llm(model_family, board_state, prompt_path)
    return moves


# Legacy
def call_gpt(board_state, prompt_path=None):
    """GPT 모델 전용 호출 함수"""
    return choose_move_with_prompt_gpt(board_state, prompt_path)

# def call_claude(board_state, prompt_path=None):
#     """Claude 모델 전용 호출 함수"""
#     return choose_move_with_prompt_claude(board_state, prompt_path)

def call_gemini(board_state, prompt_path=None):
    """Gemini 모델 전용 호출 함수"""
    return choose_move_with_prompt_gemini(board_state, prompt_path)


# Wrapper / Helper Function
def get_moves_only(model_family, board_state, prompt_path=None):
    """
    A convenience function to use when only move commands are needed.
    
    Args:
        model_family (str): One of "gpt", "claude", or "gemini".
        board_state: The current state of the board.
        prompt_path (str, optional): Path to the prompt template file.
        
    Returns:
        List: A list of parsed move commands.
    """
    _, _, _, moves = call_llm(model_family, board_state, prompt_path)
    return moves


# Batch processing function for experiments
def compare_prompts(model_family, board_state, prompt_paths):
    """
    Runs comparison experiments on the same board state using multiple prompts.
    
    Args:
        model_family (str): The model to use.
        board_state: The current state of the board.
        prompt_paths (List[str]): A list of prompt file paths to compare.
        
    Returns:
        List[dict]: A list of result dictionaries, one for each prompt tested.
    """
    results = []
    for prompt_path in prompt_paths:
        try:
            system, user, raw, moves = call_llm(model_family, board_state, prompt_path)
            results.append({
                'prompt_path': prompt_path,
                'system_prompt': system,
                'user_prompt': user,
                'raw_response': raw,
                'moves': moves,
                'move_count': len(moves),
                'success': len(moves) > 0
            })
        except Exception as e:
            results.append({
                'prompt_path': prompt_path,
                'error': str(e),
                'success': False
            })
    return results
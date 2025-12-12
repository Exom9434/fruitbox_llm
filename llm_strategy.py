# llm_strategy.py

"""
Handles all LLM (Large Language Model) interactions for generating game moves.

This module provides the core logic for calling different LLM APIs 
(OpenAI GPT and Google Gemini) to determine optimal moves based on a 
given board state.

It defines the specific data structures (using Pydantic) that the LLMs
are expected to return, ensuring validated and structured output for various
prompting strategies (e.g., Chain of Thought, per-step reasoning, judge verdicts).

Key Components:
- Pydantic Models: Defines various output formats like `Moves`, 
  `MovesWithCoT`, `MovesWithPerStepReason`, and `JudgeVerdict`.
- LLM API Functions: 
    - `choose_move_with_prompt_gpt`: Calls the OpenAI API with JSON mode.
    - `choose_move_with_prompt_gemini`: Calls the Gemini API, supporting
      JSON mode via a flattened schema.
- Utility Functions:
    - `get_pydantic_v2_flat_schema`: A helper to prepare Pydantic V2 schemas
      for compatibility with the Gemini API (removes '$defs' and 'title').
    - `load_prompt_template`: Loads prompt text from files.
"""

import json
from openai import OpenAI
import google.generativeai as genai
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Any, Tuple, Optional, Union
from google.generativeai.types import GenerationConfig, HarmCategory
from google.generativeai import ChatSession
from config import (
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    GOOGLE_API_KEY,
    MODEL_CONFIG
)
import pydantic
from typing import Dict, Any, List
import json
from openai import OpenAI
import google.generativeai as genai
import anthropic # pip install anthropic 필요
from pydantic import BaseModel, ValidationError
from typing import List, Dict, Any, Tuple, Optional
from config import MODEL_CONFIG

def get_pydantic_v2_flat_schema(model: pydantic.BaseModel) -> Dict[str, Any]:
    """
    Removes '$defs' and 'title' from a Pydantic V2 schema to create a flat structure.
    """
    # 1. Generate the base schema from the Pydantic model
    schema = model.model_json_schema()

    # 2. If '$defs' exists, flatten it to resolve references
    if '$defs' in schema:
        definitions = schema.pop('$defs')
        def resolve_refs(obj):
            if isinstance(obj, dict):
                if '$ref' in obj:
                    ref_name = obj['$ref'].split('/')[-1]
                    return resolve_refs(definitions[ref_name].copy())
                else:
                    return {k: resolve_refs(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [resolve_refs(item) for item in obj]
            else:
                return obj
        schema = resolve_refs(schema)

    # 3. Recursively remove all 'title' fields (new logic)
    def remove_titles(d):
        if isinstance(d, dict):
            d.pop('title', None)  # Remove the 'title' key if it exists, otherwise do nothing
            for v in d.values():
                remove_titles(v)
        elif isinstance(d, list):
            for i in d:
                remove_titles(i)
        return d
    
    # 4. Finally, return the schema with titles removed
    return remove_titles(schema)
# ===================================================================
# Pydantic Models
# ===================================================================

# --- Basic Moves Objects ---
class MoveObject(BaseModel):
    id: int = Field(..., description="The sequential ID of the move, starting from 1 for each turn.")
    move: List[List[Any]] = Field(..., description="A pair of coordinates for the move.")

class MoveObjectWithReason(BaseModel):
    id: int = Field(..., description="The sequential ID of the move, starting from 1 for each turn.")
    reason: str = Field(..., description="The reasoning for selecting this specific move.")
    move: List[List[Any]] = Field(..., description="A pair of coordinates for the move.")

# --- C1/C3용 점진적 생성을 위한 래퍼 모델들 ---
class SingleMoveWrapper(BaseModel):
    """A wrapper for a single move inside a target_commands list."""
    target_commands: List[MoveObject] = Field(..., description="A list containing exactly one move object.")

class SingleMoveWithReasonWrapper(BaseModel):
    """A wrapper for a single move with reasoning, inside a target_commands list."""
    target_commands: List[MoveObjectWithReason] = Field(..., description="A list containing exactly one move object with its reasoning.")

# Simple model that matches A-series prompt format  
class SimpleMoves(BaseModel):
    """A simple list of coordinate pairs matching the A-series prompt format."""
    target_commands: List[List[List[int]]] = Field(..., description="A list of coordinate pair moves, e.g., [[[1,2],[1,4]], [[3,5],[5,5]]]")

class Moves(BaseModel):
    """A list of moves, each with an ID and coordinate data."""
    target_commands: List[MoveObject] = Field(..., description="A list of move objects.")

class SingleMove(BaseModel):
    move: List[List[Any]] = Field(..., description="A single move with a pair of coordinates.")
    
# --- B-1: Chain of Thought (전체 Reason) 모델 ---
class MovesWithCoT(BaseModel):
    """A reasoning process and a list of moves."""
    reason: str = Field(..., description="The reasoning process for how the moves were selected, as a single string.") # ✅ str 타입으로 변경
    target_commands: List[MoveObject] = Field(..., description="A list of move objects derived from the reasoning.")

class JudgeVerdict(BaseModel):
    """A verdict from the Judge LLM, which is either 'OK' or 'Retry'."""
    verdict: str = Field(..., description="The judge's final decision, should be 'OK' or 'Retry'.")

# --- B-2: Reason for Each Step (개별 Reason) 모델 ---
class MoveObjectWithReason(BaseModel):
    """A single move with its own reasoning."""
    id: int = Field(..., description="The sequential ID of the move.")
    reason: str = Field(..., description="The reasoning for selecting this specific move.")
    move: List[List[Any]] = Field(..., description="A pair of coordinates for the move.")

class MovesWithPerStepReason(BaseModel):
    """A list of moves, where each move includes its own reasoning."""
    target_commands: List[MoveObjectWithReason] = Field(..., description="A list of move objects, each with an ID, reason, and coordinates.")

import json
from openai import OpenAI
import google.generativeai as genai
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Any, Tuple, Optional, Union
from google.generativeai.types import GenerationConfig, HarmCategory
from google.generativeai import ChatSession
from config import (
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    GOOGLE_API_KEY,
    MODEL_CONFIG
)
import pydantic
from typing import Dict, Any, List

# ===================================================================
# LLM Clients and Functions
# ===================================================================

# --- Client Instances ---
openai_client = OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel(MODEL_CONFIG["gemini"]["model_name"])

def load_prompt_template(path):
    """Loads a prompt template from a file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def choose_move_with_prompt_gpt(
    board_state: list,
    prompt_path: Optional[str],
    history: Optional[List[Dict[str, str]]] = None,
    response_model: Optional[BaseModel] = Moves,
    full_prompt_override: Optional[str] = None
) -> Tuple[str, str, str, Any, List[Dict[str, str]]]:
    """
    Calls the GPT API using JSON mode and Pydantic validation.
    
    This function prioritizes `full_prompt_override` as the system message if provided
    (typically for "Judge" evaluations). Otherwise, it builds prompts using `prompt_path`.

    Args:
        board_state: The current state of the board.
        prompt_path: Path to the prompt template file (used if override is not set).
        history: Conversation history to append to.
        response_model: The Pydantic model to validate the JSON response.
        full_prompt_override: If provided, this string is used directly as the 
                              system prompt, bypassing `prompt_path` and `history`.

    Returns:
        A tuple containing:
            - system_prompt (str)
            - user_prompt (str)
            - raw_response (str)
            - parsed_result (Any): The validated Pydantic object or raw response.
            - updated_history (List[Dict[str, str]])
    """
    try:
        system_prompt = ""
        user_prompt = ""

        # --- Branching Logic ---
        if full_prompt_override:
            # For Judge calls: Use the provided full prompt as the system instruction.
            system_prompt = full_prompt_override
            # A simple message to elicit a response from the model.
            user_prompt = "Based on the system instructions, provide your verdict now."
            # The Judge is a single-turn evaluation, so we build new messages, ignoring history.
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        elif prompt_path:
            # For standard calls: Separate system and user prompts.
            system_prompt = load_prompt_template(prompt_path)
            board_json = json.dumps(board_state, ensure_ascii=False)
            user_prompt = f"Here is the current game board. Find the optimal moves.\n\n## Game Board\n```json\n{board_json}\n```"
            # Construct messages based on whether history exists.
            if history:
                messages = history + [{"role": "user", "content": user_prompt}]
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
        else:
            raise ValueError("Either prompt_path or full_prompt_override must be provided.")
            
        response = openai_client.chat.completions.create(
            model=MODEL_CONFIG["gpt"]["model_name"],
            max_tokens=16384,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw_response = response.choices[0].message.content
        updated_history = messages + [{"role": "assistant", "content": raw_response}]

        # Parse results
        parsed_result = [] # Default value
        if response_model:
            try:
                parsed_data = response_model.model_validate_json(raw_response)
                parsed_result = getattr(parsed_data, 'target_commands', parsed_data)
            except ValidationError as e:
                print(f"⚠️ GPT Pydantic validation failed: {e}")
                parsed_result = []
        else:
            parsed_result = raw_response

        return system_prompt, user_prompt, raw_response, parsed_result, updated_history

    except Exception as e:
        print(f"⚠️ GPT API call failed: {e}")
        return "", "", f"API Call Failed: {e}", [], history

# ===================================================================
# 1. OpenAI Compatible Function (GPT, DeepSeek, Llama/Ollama)
# ===================================================================
def choose_move_with_openai_compatible(
    client: OpenAI, # 클라이언트를 인자로 받음 (유연성 확보)
    model_name: str,
    board_state: list,
    prompt_path: Optional[str],
    history: Optional[List[Dict[str, str]]] = None,
    response_model: Optional[BaseModel] = None, # Default None for logic
    full_prompt_override: Optional[str] = None
) -> Tuple[str, str, str, Any, List[Dict[str, str]]]:
    
    try:
        system_prompt = ""
        user_prompt = ""
        
        # 프롬프트 구성 로직 (기존과 동일)
        if full_prompt_override:
            system_prompt = full_prompt_override
            user_prompt = "Based on the system instructions, provide your verdict now."
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        elif prompt_path:
            system_prompt = load_prompt_template(prompt_path)
            board_json = json.dumps(board_state, ensure_ascii=False)
            user_prompt = f"Here is the current game board. Find the optimal moves.\n\n## Game Board\n```json\n{board_json}\n```"
            if history:
                messages = history + [{"role": "user", "content": user_prompt}]
            else:
                messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        else:
            raise ValueError("Prompt required")

        # API 호출
        response = client.chat.completions.create(
            model=model_name,
            max_tokens=4096, # 모델에 따라 조절 필요 (DeepSeek은 8k 가능)
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"}, # DeepSeek, GPT 모두 지원
        )
        raw_response = response.choices[0].message.content
        updated_history = messages + [{"role": "assistant", "content": raw_response}]

        # Pydantic 파싱
        parsed_result = []
        if response_model:
            try:
                parsed_data = response_model.model_validate_json(raw_response)
                parsed_result = getattr(parsed_data, 'target_commands', parsed_data)
            except ValidationError as e:
                print(f"⚠️ Validation failed: {e}")
                parsed_result = []
        else:
            parsed_result = raw_response

        return system_prompt, user_prompt, raw_response, parsed_result, updated_history

    except Exception as e:
        print(f"⚠️ OpenAI-Compatible API call failed: {e}")
        return "", "", f"Error: {e}", [], history

# ===================================================================
# 2. Claude (Anthropic) Function
# ===================================================================
def choose_move_with_prompt_claude(
    client: anthropic.Anthropic,
    model_name: str,
    board_state: list,
    prompt_path: Optional[str],
    history: Optional[List[Dict[str, str]]] = None,
    response_model: Optional[BaseModel] = None,
    full_prompt_override: Optional[str] = None
) -> Tuple[str, str, str, Any, List[Dict[str, str]]]:
    
    try:
        system_prompt = ""
        user_msg_content = ""

        if full_prompt_override:
            system_prompt = full_prompt_override
            user_msg_content = "Based on the system instructions, provide your verdict now."
        elif prompt_path:
            system_prompt = load_prompt_template(prompt_path)
            # Claude에게 JSON 강제화를 위한 명시적 힌트 추가
            system_prompt += "\n\nIMPORTANT: Output ONLY valid JSON."
            board_json = json.dumps(board_state, ensure_ascii=False)
            user_msg_content = f"Here is the current game board.\n\n## Game Board\n```json\n{board_json}\n```"

        # History 변환 (OpenAI format -> Claude format)
        # Claude는 System prompt가 별도 파라미터로 빠짐
        messages = []
        if history:
            # history 포맷이 openai 스타일이라고 가정하고 변환
            for msg in history:
                if msg['role'] != 'system':
                    messages.append({"role": msg['role'], "content": msg['content']})
        
        messages.append({"role": "user", "content": user_msg_content})

        response = client.messages.create(
            model=model_name,
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
            temperature=0.0
        )
        
        raw_response = response.content[0].text
        
        # History 업데이트 (OpenAI 스타일로 저장해둠 - 호환성 위해)
        updated_history = []
        if history: updated_history = history.copy()
        if not history: updated_history.append({"role": "system", "content": system_prompt})
        updated_history.append({"role": "user", "content": user_msg_content})
        updated_history.append({"role": "assistant", "content": raw_response})

        # Pydantic 파싱
        parsed_result = []
        if response_model:
            try:
                parsed_data = response_model.model_validate_json(raw_response)
                parsed_result = getattr(parsed_data, 'target_commands', parsed_data)
            except ValidationError as e:
                print(f"⚠️ Claude Validation failed: {e}")
                parsed_result = []
        else:
            parsed_result = raw_response

        return system_prompt, user_msg_content, raw_response, parsed_result, updated_history

    except Exception as e:
        print(f"⚠️ Claude API call failed: {e}")
        return "", "", f"Error: {e}", [], history

        
def choose_move_with_prompt_gemini(
    board_state: list,
    prompt_path: Optional[str],
    chat_session: Optional[genai.ChatSession] = None,
    response_model: Optional[BaseModel] = Moves,
    full_prompt_override: Optional[str] = None
) -> Tuple[str, str, str, Any, Optional[genai.ChatSession]]:
    """
    Calls the Gemini API using JSON mode (if response_model is provided) and Pydantic validation.
    
    This function prioritizes `full_prompt_override` as the system instruction if provided
    (typically for "Judge" evaluations). It also manages and returns a genai.ChatSession.

    Args:
        board_state: The current state of the board.
        prompt_path: Path to the prompt template file (used if override is not set).
        chat_session: An existing genai.ChatSession to continue the conversation.
        response_model: The Pydantic model to validate the JSON response.
        full_prompt_override: If provided, this string is used directly as the 
                              system instruction, bypassing `prompt_path`.

    Returns:
        A tuple containing:
            - system_instruction (str)
            - user_prompt_for_log (str)
            - raw_response (str)
            - parsed_result (Any): The validated Pydantic object or raw response.
            - chat_session (Optional[genai.ChatSession]): The updated chat session.
    """
    try:
        system_instruction = ""
        message_to_send = ""
        user_prompt_for_log = "" 

        if full_prompt_override:
            system_instruction = full_prompt_override
            message_to_send = "Based on the system instructions, provide your verdict now."
            user_prompt_for_log = message_to_send
        elif prompt_path:
            system_instruction = load_prompt_template(prompt_path)
            board_json = json.dumps(board_state, ensure_ascii=False)
            message_to_send = f"Here is the current game board. Find the optimal moves.\n\n## Game Board\n```json\n{board_json}\n```"
            user_prompt_for_log = message_to_send
        else:
            raise ValueError("Either prompt_path or full_prompt_override must be provided.")

         # 1. Model setup
        model = genai.GenerativeModel(
            MODEL_CONFIG["gemini"]["model_name"],
            system_instruction=system_instruction
        )
        
        response_schema_data = None
        if response_model:
            response_schema_data = get_pydantic_v2_flat_schema(response_model)
                
        generation_config = genai.GenerationConfig(
            temperature=0.0,
            max_output_tokens=16384,
            response_mime_type="application/json" if response_model else "text/plain",
            response_schema=response_schema_data 
        )
        
        safety_settings = {
            'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
            'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
            'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
            'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
        }

        # 2. API call
        if chat_session is None:
            chat_session = model.start_chat()

        response = chat_session.send_message(
            message_to_send,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        raw_response = response.text

        # 3. Parse results
        parsed_result = []
        if response_model:
            try:
                # Using model_validate_json is standard in Pydantic V2.
                parsed_data = response_model.model_validate_json(raw_response)
                parsed_result = getattr(parsed_data, 'target_commands', parsed_data)
            except (ValidationError, json.JSONDecodeError) as e:
                print(f"⚠️ Gemini Pydantic/JSON validation failed: {e}")
                parsed_result = []
        else:
            parsed_result = raw_response

        return system_instruction, user_prompt_for_log, raw_response, parsed_result, chat_session
    
    except Exception as e:
        print(f"⚠️ Gemini API call failed: {e}")
        return "", "", f"API Call Failed: {e}", [], chat_session
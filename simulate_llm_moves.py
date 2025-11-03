# simulate_llm_moves.py
"""Real Game Play Scripts"""

import os
import json
from datetime import datetime
import traceback
from typing import Any, Dict, Optional
from pydantic import BaseModel
from llm_client import call_llm
from llm_strategy import Moves, MovesWithCoT, MovesWithPerStepReason, SingleMoveWrapper, SingleMoveWithReasonWrapper, JudgeVerdict,load_prompt_template
from prompt_strategy import PromptStrategyDetector, is_a_series, is_b_series, is_c_series, is_d_series, is_e_series, get_response_model_for_prompt, get_strategy_name
from simulation import FruitBoxSimulator
from config import MODEL_CONFIG
from utils import parse_moves_from_llm
import sys
import pydantic

# --- Simulation Settings ---
BOARDS_FILE = "boards/00.json"
PROMPTS_FOLDER = "prompts_polished/A"
LOGS_FOLDER = "simulation_logs"

MODELS_TO_TEST = ["gemini"]
MAX_BOARD_ID_TO_TEST = 5
MAX_CONSECUTIVE_FAILS = 0

USE_MULTI_TURN = False
MAX_MULTI_TURNS = 4
MAX_SINGLE_TRIES = 1

EXECUTION_MODE = "real"
MAX_TURNS_TEST = 1
MAX_TURNS_REAL = 4

    
def get_max_turns():
    """Gets the maximum number of turns based on the execution mode."""
    if not USE_MULTI_TURN:
        return 100 # Allow many turns for single-turn (batch) mode
    if EXECUTION_MODE.lower() == "test":
        return MAX_TURNS_TEST
    elif EXECUTION_MODE.lower() == "real":
        return MAX_TURNS_REAL
    else:
        print(f"⚠️ Unknown EXECUTION_MODE: {EXECUTION_MODE}. Defaulting to 1 turn.")
        return 1
    
## Environment and Test Setup
print("--- [Final Environment Diagnostics Start] ---")
print(f" Python Executable Path: {sys.executable}")
print(f"📦 Pydantic Version Used: {pydantic.__version__}")
if USE_MULTI_TURN:
    print(f" Multi-Turn Mode: {EXECUTION_MODE} (Max {get_max_turns()} turns)")
else:
    print(f" Single-Turn (Batch) Mode (Max {get_max_turns()} turns)")
print("--- Python Path ---")
print(json.dumps(sys.path, indent=2))
print("--- Diagnostics End ---\n")

def setup_directories(main_log_dir, prompt_path, model):
    prompt_name = os.path.splitext(os.path.basename(prompt_path))[0]
    base_path = os.path.join(main_log_dir, prompt_name, model)
    success_path = os.path.join(base_path, "success")
    error_path = os.path.join(base_path, "error")
    os.makedirs(success_path, exist_ok=True)
    os.makedirs(error_path, exist_ok=True)
    return success_path, error_path

class PydanticEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        return json.JSONEncoder.default(self, obj)

def save_log(folder, filename, data):
    log_path = os.path.join(folder, filename)
    with open(log_path, "w", encoding="utf-8-sig") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, cls=PydanticEncoder)

def run_single_trial(board_id, board_state, model_family, prompt_path, greedy_result):
    """
    Runs a single simulation trial and returns its success status and result data.
    """
    try:
        strategy = PromptStrategyDetector.detect_strategy(prompt_path)
        if strategy:
            print(f"   - (Strategy: {strategy.strategy_name})")
            response_schema_to_use = strategy.response_model
        else:
            print("   - (Strategy: Default - Moves)")
            response_schema_to_use = Moves

        system_prompt, user_prompt, raw_response, llm_string_moves, _ = call_llm(
            model_family,
            board_state,
            prompt_path=prompt_path,
            response_model=response_schema_to_use
        )

        # Check for real API/connection failures
        if raw_response is None or "API Call Failed" in str(raw_response):
            error_log = {"error_type": "APIFailure", "message": "LLM API call failed.", "board_id": board_id, "model": model_family, "prompt_path": prompt_path, "system_prompt": system_prompt, "user_prompt": user_prompt, "llm_raw_response": raw_response}
            return False, error_log

        # Check if response is valid JSON with correct structure
        try:
            import json
            parsed_json = json.loads(raw_response)
            if not isinstance(parsed_json, dict) or 'target_commands' not in parsed_json:
                error_log = {"error_type": "InvalidJSONStructure", "message": "LLM response missing required 'target_commands' field.", "board_id": board_id, "model": model_family, "prompt_path": prompt_path, "system_prompt": system_prompt, "user_prompt": user_prompt, "llm_raw_response": raw_response}
                return False, error_log
        except json.JSONDecodeError:
            error_log = {"error_type": "InvalidJSON", "message": "LLM response is not valid JSON.", "board_id": board_id, "model": model_family, "prompt_path": prompt_path, "system_prompt": system_prompt, "user_prompt": user_prompt, "llm_raw_response": raw_response}
            return False, error_log

        # If llm_string_moves is empty, it could be legitimate (no valid moves found)
        # We should still try to parse and simulate, even with zero moves
        if llm_string_moves is None:
            llm_string_moves = []

        numeric_moves = parse_moves_from_llm(llm_string_moves)
        
        # Check if LLM found any valid moves - empty moves are considered failure
        if not numeric_moves:
            error_log = {"error_type": "NoMovesFound", "message": "LLM returned valid JSON but found no moves.", "board_id": board_id, "model": model_family, "prompt_path": prompt_path, "system_prompt": system_prompt, "user_prompt": user_prompt, "llm_raw_response": raw_response}
            return False, error_log

        simulator = FruitBoxSimulator(board_state)
        simulator.apply_moves(numeric_moves)
        simulation_result = simulator.get_result()

        success_log = {
            "board_id": board_id, "model": model_family, "prompt_file": os.path.basename(prompt_path), "initial_board_state": board_state,
            "llm_result": {"moves": llm_string_moves, "simulation_result": simulation_result},
            "comparison_baselines": {"greedy": greedy_result},
            "system_prompt": system_prompt, "user_prompt": user_prompt, "llm_raw_response": raw_response
        }
        return True, success_log

    except Exception as e:
        error_log = {"error_type": "CriticalSimulationError", "message": str(e), "traceback": traceback.format_exc(), "board_id": board_id, "model": model_family, "prompt_path": prompt_path}
        return False, error_log

def run_multi_turn_session(board_id, board_state, model_family, prompt_path, greedy_result):
    history = None
    conversation_log = []
    max_turns = get_max_turns()
    simulator = FruitBoxSimulator(board_state)  # For tracking board state
    all_found_moves = []
    turn_details = []
    
    strategy_name = get_strategy_name(prompt_path)
    print(f"   - (Strategy: {strategy_name})")

    for turn in range(1, max_turns + 1):
        print(f"   - Turn #{turn}/{max_turns}...", end="")
        
        turn_info = {
            "turn": turn,
            "board_state_before": simulator.board.tolist(),
            "score_before": simulator.score
        }
        
        try:
            system_prompt, user_prompt, raw_response, llm_moves, updated_history = call_llm(
                model_family, simulator.board.tolist(), prompt_path, history=history  # USE the updated board
            )
            history = updated_history
            turn_info["raw_response"] = raw_response
            conversation_log.append({ "turn": turn, "response": raw_response, "parsed_moves": llm_moves })

            if llm_moves:
                print(" ✅ Success!")
                score_before_moves = simulator.score
                valid_moves = []
                
                # Apply each move sequentially
                for move in llm_moves:
                    if simulator.apply_move(move):
                        valid_moves.append(move)
                        all_found_moves.append(move)
                
                score_gain = simulator.score - score_before_moves
                turn_info.update({
                    "status": "success",
                    "attempted_moves": llm_moves,
                    "valid_moves": valid_moves,
                    "score_after": simulator.score,
                    "score_gain": score_gain,
                    "board_state_after": simulator.board.tolist()
                })
                turn_details.append(turn_info)
                
                # If valid moves exist, continue (C3/C4 run for multiple turns)
                if len(valid_moves) == 0:
                    print(" (No valid moves in practice)")
                    
            else:
                print(" ❌ (No valid moves found)")
                turn_info["status"] = "no_moves_found"
                turn_details.append(turn_info)
                
        except Exception as e:
            print(f" ❌ Critical Error!")
            turn_info["status"] = "error"
            turn_info["error"] = str(e)
            turn_details.append(turn_info)
            return False, {"error_type": "CriticalSessionError", "message": str(e), "turn": turn, "turn_details": turn_details}

    final_result = simulator.get_result()
    print(f"   - {strategy_name} session finished. Found {len(all_found_moves)} total moves. Final score: {final_result['score']}")
    
    success = len(all_found_moves) > 0
    return success, {
        "board_id": board_id, 
        "strategy": strategy_name,
        "status": "Success" if success else "Failure", 
        "total_turns": max_turns,
        "llm_result": {"moves": all_found_moves, "simulation_result": final_result},
        "conversation_history": conversation_log,
        # Add turn-by-turn details
        "turn_details": turn_details  
    }

def run_simulation_experiment():
    test_start_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    main_log_dir = os.path.join(LOGS_FOLDER, f"test_{test_start_time}")
    print(f"📄 All logs will be saved to the '{main_log_dir}' directory.")

    try:
        with open(BOARDS_FILE, "r", encoding="utf-8-sig") as f:
            all_board_data = json.load(f)
        print(f"✅ Loaded {len(all_board_data)} board data entries from '{BOARDS_FILE}'.")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ Error: Failed to open '{BOARDS_FILE}'. ({e})")
        return

    try:
        prompt_files = [os.path.join(PROMPTS_FOLDER, f) for f in os.listdir(PROMPTS_FOLDER) if f.endswith(".txt")]
        prompt_files.sort()
        if not prompt_files: raise FileNotFoundError
        print(f"✅ Found {len(prompt_files)} prompts in '{PROMPTS_FOLDER}'.")
    except FileNotFoundError:
        print(f"❌ Error: No .txt prompt files found in '{PROMPTS_FOLDER}'.")
        return

    for board_data in all_board_data:
        board_id = board_data.get("id", "N/A")
        board_state = board_data.get("board")

        if not isinstance(board_id, int):
            print(f"⚠️ Board ID '{board_id}' is not an integer, skipping.")
            continue

        if MAX_BOARD_ID_TO_TEST is not None and board_id > MAX_BOARD_ID_TO_TEST:
            print(f"\n✅ Target Board ID ({MAX_BOARD_ID_TO_TEST}) reached. Halting tests.")
            break

        if not board_state:
            print(f"⚠️ No board data found for Board ID {board_id}, skipping.")
            continue

        for prompt_path in prompt_files:
            prompt_name = os.path.basename(prompt_path)
            if "generator.txt" in prompt_name:
                print(f"⏩ Skipping '{prompt_name}' as it is called by the Judge strategy.")
                continue

            print(f"\n--- [Board ID: {board_id} | Prompt: {prompt_name}] ---")

            print(f"🤖 Running Greedy algorithm...")
            greedy_score, greedy_moves = FruitBoxSimulator.greedy_simulation(board_state)
            greedy_result = {"score": greedy_score, "moves": greedy_moves, "moves_cnt": len(greedy_moves)}
            print(f"   - Greedy result: Score = {greedy_score}, Moves = {len(greedy_moves)}")

            for model_family in MODELS_TO_TEST:
                print(f"\n🚀 Starting experiment: [Model: {model_family.upper()}] [Mode: {'Multi-Turn' if USE_MULTI_TURN else 'Single-Turn'}]")
                success_path, error_path = setup_directories(main_log_dir, prompt_path, model_family)
                
                is_success = False
                result_data = {}
                
                strategy = PromptStrategyDetector.detect_strategy(prompt_path)

                # [Modified] Simplified strategy execution logic to run only once.
                if strategy is None:
                    print(f"⚠️ Unknown prompt strategy: {prompt_name}. Running default single trial.")
                    is_success, result_data = run_single_trial(board_id, board_state, model_family, prompt_path, greedy_result)
                elif is_a_series(prompt_path) or is_b_series(prompt_path) or is_e_series(prompt_path):
                    is_success, result_data = run_single_trial(board_id, board_state, model_family, prompt_path, greedy_result)
                elif is_c_series(prompt_path):
                    if USE_MULTI_TURN:
                        if strategy.sub_strategy.value == "C1":
                            is_success, result_data = run_incremental_session(board_id, board_state, model_family, prompt_path, greedy_result, response_model=strategy.response_model, strategy_name=strategy.strategy_name)
                        elif strategy.sub_strategy.value == "C2":
                            is_success, result_data = run_c2_batch_session(board_id, board_state, model_family, prompt_path, greedy_result)
                        elif strategy.sub_strategy.value == "C3":
                            is_success, result_data = run_multi_turn_session(board_id, board_state, model_family, prompt_path, greedy_result)
                        else:
                            is_success, result_data = run_multi_turn_session(board_id, board_state, model_family, prompt_path, greedy_result)
                    else: # If multi-turn is disabled, run as single-turn instead
                        is_success, result_data = run_single_trial(board_id, board_state, model_family, prompt_path, greedy_result)
                elif is_d_series(prompt_path):
                    if USE_MULTI_TURN:
                        if strategy.sub_strategy.value == "D1":
                            is_success, result_data = run_d1_judge_session(board_id, board_state, model_family, prompt_path, greedy_result)
                        else:
                            is_success, result_data = run_multi_turn_session(board_id, board_state, model_family, prompt_path, greedy_result)
                    else: # If multi-turn is disabled, run as single-turn instead
                        is_success, result_data = run_single_trial(board_id, board_state, model_family, prompt_path, greedy_result)
                else:
                    is_success, result_data = run_single_trial(board_id, board_state, model_family, prompt_path, greedy_result)

                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                if is_success:
                    print(f" ✅ Success!")
                    filename = f"board_{board_id}_success_{timestamp}.json"
                    save_log(success_path, filename, result_data)
                else:
                    # [Modified] Simplified failure logic and removed redundant execution code.
                    print(f" ❌ Failure")
                    filename = f"board_{board_id}_fail_{timestamp}.json"
                    save_log(error_path, filename, result_data)
                    # Logic to halt on consecutive failures can be added here if needed.
                    # e.g.: if consecutive_fails >= MAX_SINGLE_TRIES: break

    print("\n\n All simulations completed.")

# def run_incremental_session(board_id, board_state, model_family, prompt_path, greedy_result, response_model=None, strategy_name=None):
#     if strategy_name is None:
#         strategy_name = get_strategy_name(prompt_path)
#     if response_model is None:
#         response_model = get_response_model_for_prompt(prompt_path)

#     print(f"   - (Strategy: {strategy_name} Incremental Generation)")

#     conversation_history = None
#     all_found_moves = []
#     simulator = FruitBoxSimulator(board_state)
#     max_turns = get_max_turns()
#     final_turn = 0
#     turn_details = []  # Store turn-by-turn details

#     for turn in range(1, max_turns + 1):
#         final_turn = turn
#         print(f"     - {strategy_name} Turn #{turn}: Searching for the optimal single move on the current board...", end="")
        
#         turn_info = {
#             "turn": turn,
#             "board_state_before": simulator.board.tolist(),
#             "score_before": simulator.score
#         }

#         try:
#             _, _, raw_response, parsed_move_list, updated_history = call_llm(
#                 model_family,
#                 simulator.board.tolist(),
#                 prompt_path,
#                 history=conversation_history,
#                 response_model=response_model
#             )
#             conversation_history = updated_history
#             turn_info["raw_response"] = raw_response

#             # Terminate only if the LLM explicitly declares completion
#             if "no more moves" in raw_response.lower():
#                 print(" ➡️ LLM declared completion.")
#                 turn_info["status"] = "llm_declared_end"
#                 turn_details.append(turn_info)
#                 break

#             # If no moves are parsed, continue to the next turn
#             if not parsed_move_list:
#                 print(" ❌ No valid moves, trying next turn")
#                 turn_info["status"] = "no_moves_parsed"
#                 turn_details.append(turn_info)
#                 continue

#             move_object = parsed_move_list[0]
#             single_move_coords = move_object.move
#             turn_info["attempted_move"] = single_move_coords
            
#             score_before = simulator.score
#             move_valid = simulator.apply_move(single_move_coords)

#             if move_valid:
#                 score_gain = simulator.score - score_before
#                 print(f" ✅ Move found! (Score +{score_gain})")
#                 all_found_moves.append(single_move_coords)
#                 turn_info.update({
#                     "status": "success",
#                     "score_after": simulator.score,
#                     "score_gain": score_gain,
#                     "board_state_after": simulator.board.tolist()
#                 })
#             else:
#                 print(f" ⚠️ Invalid move, trying next turn")
#                 turn_info["status"] = "invalid_move"

#         except Exception as e:
#             print(f" ❌ Error occurred ({str(e)}), trying next turn")
#             turn_info["status"] = "error"
#             turn_info["error"] = str(e)
            
#         turn_details.append(turn_info)
            
#     final_result = simulator.get_result()
#     print(f"   - {strategy_name} session finished. Found {len(all_found_moves)} total moves. Final score: {final_result['score']}")

#     return len(all_found_moves) > 0, {
#         "board_id": board_id, "strategy": strategy_name,
#         "llm_result": {"moves": all_found_moves, "simulation_result": final_result},
#         "total_turns": final_turn,
#         "turn_details": turn_details  # Add turn-by-turn details
#     }
    
# def run_d1_judge_session(board_id, board_state, model_family, prompt_path, greedy_result):
#     strategy_name = get_strategy_name(prompt_path)
#     print(f"   - (전략: {strategy_name})")

#     generator_history = None
#     judge_history = None
#     generator_model = model_family
#     judge_model = model_family
#     judge_conversation_log = []
    
#     # Get paths for generator and judge prompts
#     base_dir = os.path.dirname(prompt_path)
#     generator_prompt_path = os.path.join(base_dir, 'D1_generator.txt')
#     judge_prompt_path = os.path.join(base_dir, 'D1_judge.txt')
    
#     for attempt in range(1, 6):
#         print(f"     - Judge 시도 #{attempt}/5")

#         # Use generator prompt file
#         _, _, gen_raw_response, generated_moves, updated_gen_history = call_llm(
#             generator_model, board_state, generator_prompt_path, history=generator_history, response_model=None
#         )
#         generator_history = updated_gen_history
        
#         if not generated_moves:
#             print("       - Generator가 이동을 생성하지 못했습니다.")
#             continue

#         # Use judge prompt file with board state and candidate moves
#         with open(judge_prompt_path, 'r', encoding='utf-8-sig') as f:
#             judge_prompt_template = f.read()

#         judge_full_prompt = judge_prompt_template.replace(
#             "{board_state_json}", json.dumps(board_state)
#         ).replace(
#             "{candidate_moves_json}", json.dumps([move.model_dump() if hasattr(move, 'model_dump') else move for move in generated_moves])
#         )
        
#         _, _, judge_raw_response, _, updated_judge_history = call_llm(
#             judge_model, board_state=None, prompt_path=judge_prompt_path, full_prompt_override=judge_full_prompt, history=judge_history, response_model=None
#         )
#         judge_history = updated_judge_history

#         judge_interaction = { "attempt": attempt, "generated_moves": generated_moves, "judge_raw_response": judge_raw_response }
#         judge_conversation_log.append(judge_interaction)

#         # Parse judge response
#         try:
#             judge_data = json.loads(judge_raw_response)
#             verdict = judge_data.get("verdict", "").lower()
            
#             if verdict == "ok":
#                 print("       - ✅ Judge가 승인했습니다. 시뮬레이션을 시작합니다.")
                
#                 # Use approved moves if provided, otherwise use original generated moves
#                 approved_moves = judge_data.get("approved_moves", {}).get("target_commands", generated_moves)
#                 final_moves = []
                
#                 for move_item in approved_moves:
#                     if hasattr(move_item, 'move'):
#                         final_moves.append(move_item.move)
#                     elif isinstance(move_item, dict) and 'move' in move_item:
#                         final_moves.append(move_item['move'])
#                     else:
#                         final_moves.append(move_item)
                
#                 simulator = FruitBoxSimulator(board_state)
#                 simulator.apply_moves(final_moves)
#                 final_result = simulator.get_result()
                
#                 return True, {
#                     "board_id": board_id, "strategy": strategy_name,
#                     "llm_result": {"moves": final_moves, "simulation_result": final_result},
#                     "judge_conversation_log": judge_conversation_log,
#                     "total_attempts": attempt
#                 }
#             else:
#                 print("       - 🔄 Judge가 재시도를 요청했습니다.")
#                 feedback = judge_data.get("feedback", "Please find better moves.")
#                 if generator_history is None:
#                     generator_history = []
#                 generator_history.append({"role": "user", "content": f"The judge said to retry. Feedback: {feedback}"})
#         except (json.JSONDecodeError, KeyError) as e:
#             print(f"       - ⚠️ Judge 응답 파싱 실패: {e}")
#             if generator_history is None:
#                 generator_history = []
#             generator_history.append({"role": "user", "content": "The judge response was unclear. Please find better moves."})

#     print("     - ⚠️ Judge가 5회 안에 승인하지 않아 실패 처리합니다.")
#     return False, {
#         "board_id": board_id, "strategy": strategy_name,
#         "error": "Judge failed to approve within 5 attempts.",
#         "judge_conversation_log": judge_conversation_log,
#         "total_attempts": 5
#     }


# def run_c2_batch_session(board_id, board_state, model_family, prompt_path, greedy_result):
#     """
#     C2 전략: 한 번에 20개 움직임을 요청하고 4번 반복
#     각 반복마다 시뮬레이션으로 업데이트된 보드 정보를 제공
#     """
#     strategy_name = get_strategy_name(prompt_path)
#     print(f"   - (전략: {strategy_name})")
    
#     simulator = FruitBoxSimulator(board_state)
#     all_successful_moves = []
#     conversation_history = None
#     batch_logs = []
    
#     MAX_BATCHES = 4
#     MOVES_PER_BATCH = 20
    
#     for batch_num in range(1, MAX_BATCHES + 1):
#         print(f"     - C2 Batch #{batch_num}/{MAX_BATCHES}: {MOVES_PER_BATCH}개 움직임 요청 중...", end="")
        
#         try:
#             # 현재 보드 상태를 사용하여 LLM 호출
#             current_board = simulator.board.tolist()
            
#             _, _, raw_response, parsed_moves, updated_history = call_llm(
#                 model_family,
#                 current_board,
#                 prompt_path,
#                 history=conversation_history,
#                 response_model=get_response_model_for_prompt(prompt_path)
#             )
#             conversation_history = updated_history
            
#             batch_log = {
#                 "batch_number": batch_num,
#                 "board_state_before": current_board,
#                 "raw_response": raw_response,
#                 "parsed_moves": parsed_moves,
#                 "successful_moves": [],
#                 "moves_applied": 0,
#                 "score_gain": 0
#             }
            
#             if not parsed_moves:
#                 print(" ❌ 유효한 움직임 없음")
#                 batch_logs.append(batch_log)
#                 continue
                
#             print(f" 파싱됨: {len(parsed_moves)}개 항목", end="")
                
#             # 파싱된 움직임을 실제 좌표로 변환
#             moves_to_apply = []
            
#             if parsed_moves and len(parsed_moves) > 0:
#                 # parsed_moves가 리스트 형태인 경우 (직접 target_commands가 파싱된 경우)
#                 if isinstance(parsed_moves, list):
#                     for move_item in parsed_moves:
#                         if hasattr(move_item, 'move'):
#                             moves_to_apply.append(move_item.move)
#                         elif isinstance(move_item, dict) and 'move' in move_item:
#                             moves_to_apply.append(move_item['move'])
#                         else:
#                             moves_to_apply.append(move_item)
#                 # parsed_moves[0]이 target_commands를 가진 객체인 경우
#                 elif hasattr(parsed_moves[0], 'target_commands'):
#                     for step in parsed_moves[0].target_commands:
#                         if hasattr(step, 'move'):
#                             moves_to_apply.append(step.move)
#                         elif isinstance(step, dict) and 'move' in step:
#                             moves_to_apply.append(step['move'])
#                         else:
#                             moves_to_apply.append(step)
            
#             # 최대 20개까지만 적용
#             moves_to_apply = moves_to_apply[:MOVES_PER_BATCH]
#             print(f" → {len(moves_to_apply)}개 추출", end="")
            
#             # 각 움직임을 하나씩 적용하고 유효성 검사
#             successful_moves_in_batch = []
#             score_before_batch = simulator.score
            
#             for move in moves_to_apply:
#                 if simulator.apply_move(move):
#                     successful_moves_in_batch.append(move)
#                     all_successful_moves.append(move)
            
#             score_after_batch = simulator.score
#             score_gain = score_after_batch - score_before_batch
            
#             batch_log.update({
#                 "successful_moves": successful_moves_in_batch,
#                 "moves_applied": len(successful_moves_in_batch),
#                 "score_gain": score_gain,
#                 "board_state_after": simulator.board.tolist()
#             })
#             batch_logs.append(batch_log)
            
#             print(f" ✅ {len(successful_moves_in_batch)}개 적용됨 (점수 +{score_gain})")
            
#             # 유효한 움직임이 없으면 조기 종료 (단, REAL 모드에서는 계속 진행)
#             if len(successful_moves_in_batch) == 0:
#                 if EXECUTION_MODE.lower() == "real":
#                     print(f"     - 유효한 움직임이 없지만 Real 모드에서 계속 진행합니다.")
#                 else:
#                     print(f"     - 유효한 움직임이 없어 배치 반복을 종료합니다.")
#                     break
                
#         except Exception as e:
#             print(f" ❌ 에러 발생: {str(e)}")
#             batch_log = {
#                 "batch_number": batch_num,
#                 "error": str(e),
#                 "board_state_before": current_board
#             }
#             batch_logs.append(batch_log)
#             continue
    
#     final_result = simulator.get_result()
#     print(f"   - C2 세션 종료. 총 {len(all_successful_moves)}개의 수 발견. 최종 점수: {final_result['score']}")
    
#     return len(all_successful_moves) > 0, {
#         "board_id": board_id, 
#         "strategy": strategy_name,
#         "llm_result": {"moves": all_successful_moves, "simulation_result": final_result},
#         "batch_details": batch_logs,
#         "total_batches": len(batch_logs),
#         "total_moves": len(all_successful_moves)
#     }


if __name__ == "__main__":
    run_simulation_experiment()
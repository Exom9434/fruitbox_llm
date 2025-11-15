# utils.py

from math import hypot
import numpy as np
import pyautogui


def drag_move(values, r_start, c_start, r_end, c_end):
    try:

        r_min, r_max = min(r_start, r_end), max(r_start, r_end)
        c_min, c_max = min(c_start, c_end), max(c_start, c_end)

        # 1. Store information of fruits within the selected area in a list
        selected_fruits = []
        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                if values[r, c, 0] != 0: # If num (fruit ID) is not 0
                    selected_fruits.append(values[r, c])
        
        if not selected_fruits:
            print("⚠️ No valid fruits found in the selected area.")
            return

        # 2. Convert the list to a numpy array for easier calculation
        fruits_array = np.array(selected_fruits)
        # fruits_array[:, 1] contains all x_center values
        # fruits_array[:, 2] contains all y_center values

        # 3. Calculate min/max values of center points and average sizes
        min_center_x = np.min(fruits_array[:, 1])
        max_center_x = np.max(fruits_array[:, 1])
        min_center_y = np.min(fruits_array[:, 2])
        max_center_y = np.max(fruits_array[:, 2])
        
        avg_w = np.mean(fruits_array[:, 3])
        avg_h = np.mean(fruits_array[:, 4])

        # 4. Combine the center point range and average size to calculate final bounds
        #    e.g., Move left from the leftmost center point by half the average width
        min_x = min_center_x - (avg_w / 2)
        max_x = max_center_x + (avg_w / 2)
        min_y = min_center_y - (avg_h / 2)
        max_y = max_center_y + (avg_h / 2)

        # 5. Calculate final drag coordinates (with padding)
        padding = 25 # Generous padding value
        drag_start_x = int(min_x)
        drag_start_y = int(min_y) 
        drag_end_x = int(max_x) + padding
        drag_end_y = int(max_y) + padding

        # 6. Debug logging and execute drag
        print(f"💡 Center Point Range: X({min_center_x:.0f}~{max_center_x:.0f}), Y({min_center_y:.0f}~{max_center_y:.0f})")
        print(f"💡 Average Size: W({avg_w:.1f}), H({avg_h:.1f})")
        print(f"📐 Full Bounding Box: ({drag_start_x}, {drag_start_y}) -> ({drag_end_x}, {drag_end_y})")
        
        distance = hypot(drag_end_x - drag_start_x, drag_end_y - drag_start_y)
        duration = 0.8 # Fixed time or could be based on distance

        pyautogui.moveTo(drag_start_x, drag_start_y, duration=0.1)
        pyautogui.dragTo(drag_end_x, drag_end_y, duration=duration, button="left")
        
        print(f"🖱️ Drag complete (Distance {distance:.1f}px, Duration {duration:.2f}s)")

    except Exception as e:
        print(f"❌ Exception during drag: {e}")
def parse_moves_from_llm(llm_response_data: list) -> list:
    """
    Parses the LLM's response.
    
    Handles various formats (Pydantic objects, dictionaries, lists) and 
    converts them into a list of numerical coordinates usable by the simulator.
    """
    if not llm_response_data:
        return []

    parsed_moves = []
    try:
        first_item = llm_response_data[0]

        # 1. Case: List of Pydantic objects (e.g., [MoveObject(move=[...])])
        if hasattr(first_item, 'move'):
            for move_object in llm_response_data:
                command = move_object.move
                # Call the actual parsing logic
                parsed_moves.append(_parse_single_command(command))

        # 2. Case: List of dictionaries (e.g., [{'id':1, 'move':[...]}])
        elif isinstance(first_item, dict):
            for move_dict in llm_response_data:
                # Use .get() in case the 'move' key is missing
                command = move_dict.get('move')
                if command:
                    parsed_moves.append(_parse_single_command(command))
        
        # 3. Case: List of lists (e.g., [[[r1,c1],[r2,c2]]])
        elif isinstance(first_item, list):
            for command in llm_response_data:
                parsed_moves.append(_parse_single_command(command))

    except Exception as e:
        print(f" - ⚠️ Parsing Error: {e}")
        print(f"   - Original data: {llm_response_data}")
        return []

    # Filter out None values to keep only valid moves
    return [move for move in parsed_moves if move is not None]

def _parse_single_command(command: list) -> list | None:
    """Internal helper function to parse a single command."""
    try:
        # Filter out invalid command formats
        if not command or len(command) != 2:
            return None
            
        start_coord, end_coord = command

        # 1. Case: Already in numerical format
        if isinstance(start_coord[0], int):
            return command
        
        # 2. Case: String format (e.g., 'r1', 'c2')
        elif isinstance(start_coord[0], str):
            start_r = int(start_coord[0].replace('r', ''))
            start_c = int(start_coord[1].replace('c', ''))
            end_r = int(end_coord[0].replace('r', ''))
            end_c = int(end_coord[1].replace('c', ''))
            return [[start_r, start_c], [end_r, end_c]]
            
    except (ValueError, TypeError, IndexError):
        # If an error occurs during parsing, skip this command
        return None
    
    return None
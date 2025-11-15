#analyze_result.py
"""
Script to analyze the simulation Results 
"""

import os
import json
import pandas as pd
import argparse
from glob import glob

LOGS_FOLDER = "simulation_logs"

def analyze_c_series_details(data, prompt, model, c_series_detailed_data):
    """
    Analyzes detailed information for C-series strategies and appends it to
    the c_series_detailed_data list.
    """
    board_id = data.get("board_id", "N/A")
    strategy = data.get("strategy", prompt)
    
    # Common Information
    base_info = {
        "Board_ID": board_id,
        "Strategy": strategy,
        "Model": model,
        "Final_Score": data.get("llm_result", {}).get("simulation_result", {}).get("score", 0),
        "Total_Moves": len(data.get("llm_result", {}).get("moves", [])),
        "Total_Turns": data.get("total_turns", 0)
    }
    
    # C1,C3,C4 Details
    if "turn_details" in data:
        for turn_detail in data["turn_details"]:
            turn_info = base_info.copy()
            turn_info.update({
                "Turn": turn_detail.get("turn", 0),
                "Turn_Status": turn_detail.get("status", "unknown"),
                "Score_Before": turn_detail.get("score_before", 0),
                "Score_After": turn_detail.get("score_after", turn_detail.get("score_before", 0)),
                "Score_Gain": turn_detail.get("score_gain", 0),
                "Move_Type": "incremental" if strategy.startswith("C1") else "multi_turn"
            })
            c_series_detailed_data.append(turn_info)
    
    # C2 Details
    elif "batch_details" in data:
        for batch_detail in data["batch_details"]:
            batch_info = base_info.copy()
            batch_info.update({
                "Turn": batch_detail.get("batch_number", 0),
                "Turn_Status": "success" if batch_detail.get("moves_applied", 0) > 0 else "no_moves",
                "Score_Before": batch_detail.get("board_state_before", []),  
                "Score_After": batch_detail.get("board_state_after", []),
                "Score_Gain": batch_detail.get("score_gain", 0),
                "Move_Type": "batch",
                "Moves_Applied": batch_detail.get("moves_applied", 0),
                "Batch_Size": len(batch_detail.get("successful_moves", []))
            })
            c_series_detailed_data.append(batch_info)

def find_latest_log_dir(base_folder):
    """
    Finds the most recently modified 'test_*' directory within the base folder.
    """
    test_dirs = [d for d in glob(os.path.join(base_folder, "test_*")) if os.path.isdir(d)]
    if not test_dirs:
        return None
    return max(test_dirs, key=os.path.getmtime)

def analyze_logs(log_dir):
    """
    Analyze all the results in the folder and save the CSV
    """
    print(f"Starting log analysis: '{log_dir}'")

    # Find out all the json files
    all_json_files = glob(os.path.join(log_dir, "**", "*.json"), recursive=True)

    if not all_json_files:
        print(" Could not find log files (.json) to analyze.")
        return

    print(f"📄 Found a total of {len(all_json_files)} log files (success + error).")

    all_trials_data = []
    # For 'C' series
    c_series_detailed_data = []  
    
    for file_path in all_json_files:
        try:
            # Extract the Prompts, Models and the Status.
            parts = file_path.replace("\\", "/").split('/')
            if len(parts) < 4: continue

            status = parts[-2]  # 'success' or 'error'
            model = parts[-3]
            prompt = os.path.splitext(parts[-4])[0] 

            if status not in ['success', 'error']: continue

            trial_info = {"Prompt": prompt, "Model": model, "Status": status}

            # To make only Succesful ones.
            if status == 'success':
                with open(file_path, 'r', encoding='utf-8-sig') as f:
                    data = json.load(f)
                
                trial_info["LLM Score"] = data.get("llm_result", {}).get("simulation_result", {}).get("score")
                trial_info["Greedy Score"] = data.get("comparison_baselines", {}).get("greedy", {}).get("score")
                
                # For C-series
                if prompt.startswith('C'):
                    analyze_c_series_details(data, prompt, model, c_series_detailed_data)

            all_trials_data.append(trial_info)
        
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f" Error: An error occurred while processing file '{file_path}'. ({e})")

    if not all_trials_data:
        print(" No valid data to analyze.")
        return

    df = pd.DataFrame(all_trials_data)

    # --- 1. Error rate ---
    count_summary = df.groupby(['Prompt', 'Model', 'Status']).size().unstack(fill_value=0)
    
    if 'success' not in count_summary: count_summary['success'] = 0
    if 'error' not in count_summary: count_summary['error'] = 0
    
    count_summary['Total_Trials'] = count_summary['success'] + count_summary['error']
    count_summary['Error_Rate'] = (count_summary['error'] / count_summary['Total_Trials']).fillna(0)
    
    count_summary = count_summary.rename(columns={'success': 'Successful_Trials', 'error': 'Error_Trials'})

    # --- 2. Calculate "Success" Score ---
    success_df = df[df['Status'] == 'success'].copy()
    
    score_summary = pd.DataFrame()
    if not success_df.empty:
        success_df['Score Delta (LLM - Greedy)'] = success_df['LLM Score'] - success_df['Greedy Score']
        score_summary = success_df.groupby(['Prompt', 'Model']).agg(
            Avg_LLM_Score=('LLM Score', 'mean'),
            Avg_Greedy_Score=('Greedy Score', 'mean'),
            Avg_Score_Delta=('Score Delta (LLM - Greedy)', 'mean'),
            Std_Dev_Delta=('Score Delta (LLM - Greedy)', 'std'),
        )

    # --- 3. Merge the Results ---
    final_summary = count_summary.join(score_summary).fillna(0)

    display_columns = [
        'Total_Trials', 'Successful_Trials', 'Error_Trials', 'Error_Rate',
        'Avg_LLM_Score', 'Avg_Greedy_Score', 'Avg_Score_Delta', 'Std_Dev_Delta'
    ]
    final_summary = final_summary[[col for col in display_columns if col in final_summary.columns]]

    print("\n\n--- [ Overall Results Summary (Including Error Rate) ] ---\n")
    print(final_summary.to_string(formatters={
        'Error_Rate': '{:.2%}'.format,
        'Avg_LLM_Score': '{:,.2f}'.format,
        'Avg_Greedy_Score': '{:,.2f}'.format,
        'Avg_Score_Delta': '{:,.2f}'.format,
        'Std_Dev_Delta': '{:,.2f}'.format
    }))

    # --- 4. Save to CSV file ---
    output_filename = "analysis_summary.csv"
    output_path = os.path.join(log_dir, output_filename)
    try:
        final_summary.to_csv(output_path, encoding='utf-8-sig')
        print(f"\n💾 Analysis results saved to CSV file: {output_path}")
    except Exception as e:
        print(f"\n An error occurred while saving the CSV file: {e}")
    
    # --- For C Series ---
    if c_series_detailed_data:
        c_series_df = pd.DataFrame(c_series_detailed_data)
        c_series_output_path = os.path.join(log_dir, "c_series_turn_details.csv")
        try:
            c_series_df.to_csv(c_series_output_path, encoding='utf-8-sig', index=False)
            print(f"📊 C-series detailed turn analysis saved: {c_series_output_path}")
            
            # C-series summary statistics
            print("\n--- [ C-Series Turn Analysis Summary ] ---")
            c_summary = c_series_df.groupby(['Strategy', 'Model']).agg({
                'Score_Gain': ['sum', 'mean', 'count'],
                'Turn': 'max',
                'Final_Score': 'first'
            }).round(2)
            print(c_summary.to_string())
            
        except Exception as e:
            print(f"\n Error saving C-series detailed analysis: {e}")
    else:
        print("\n No C-series data found, skipping detailed analysis.")
    # -------------------------------------------
    
    print("\n\nAnalysis complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze simulation log files, including error rates, and save to CSV.")
    parser.add_argument(
        '--log_dir', 
        type=str, 
        help=f"Path to a specific log directory to analyze. If not specified, automatically finds the latest log in '{LOGS_FOLDER}'."
    )
    
    args = parser.parse_args()
    
    target_log_dir = args.log_dir
    
    if not target_log_dir:
        print(f"Searching for the latest log directory in '{LOGS_FOLDER}'...")
        target_log_dir = find_latest_log_dir(LOGS_FOLDER)
    
    if not target_log_dir or not os.path.isdir(target_log_dir):
        print(f" Error: Could not find log directory to analyze: '{target_log_dir}'")
    else:
        analyze_logs(target_log_dir)
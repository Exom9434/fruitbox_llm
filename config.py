# config.py
"""For API and Game Configuration""
from dotenv import load_dotenv
import os
from load_env import load_environment

# load env file
load_environment()

# Read env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Modeil Config
MODEL_CONFIG = {
    "gpt": {
        "model_name": "gpt-4o-2024-11-20",  
        "default_max_tokens": 8192,
    },
    "claude": {
        "model_name": "claude-sonnet-4-20250514", 
        "default_max_tokens": 8192,
    },
    "gemini": {
        "model_name": "gemini-2.5-flash", 
    }
}

ROWS = 10
COLS = 17

# Display Ratio 
PIXEL_RATIO = None  #
# Margin
MAX_MARGIN = 25

# Template Image Path
TEMPLATE_DIR = "data/"

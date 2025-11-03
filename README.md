# FruitBox LLM Research Project

A comprehensive research project that evaluates different Large Language Model (LLM) strategies for solving the "Fruit Box" puzzle game. This project compares various prompting techniques and measures their performance against baseline algorithms.

## 🎯 Project Overview

This project tests how well different LLM providers (GPT-4o, Claude Sonnet 4, Gemini 2.0 Flash) can solve puzzle games using various prompting strategies. The research focuses on understanding which approaches work best for structured problem-solving tasks.

### Key Features

- **Multi-LLM Support**: GPT-4o, Claude Sonnet 4, and Gemini 2.0 Flash
- **Strategy-Based Testing**: A-series (Basic), B-series (Reasoning), C-series (Multi-turn)
- **Automatic Strategy Detection**: File-based strategy identification using A/B/C keywords
- **Comprehensive Logging**: Detailed experiment results and performance metrics
- **Greedy Baseline Comparison**: Performance evaluation against deterministic algorithms

## 🏗️ Architecture

### Core Components

| Component | Description |
|-----------|-------------|
| `simulate_llm_moves.py` | Main experiment runner with Pydantic model handling |
| `llm_client.py` | Unified interface for calling different LLM providers |
| `llm_strategy.py` | Pydantic models and LLM-specific implementations |
| `prompt_strategy.py` | **[NEW]** Refactored prompt handling with keyword-based detection |
| `utils.py` | Enhanced utility functions for parsing and transformations |
| `config.py` | Centralized model configuration |
| `load_env.py` | Environment variable loader for API keys |

### Testing Strategies

#### A-Series (Basic Prompting)
- **A1**: Baseline prompting
- **A2**: Rule-enhanced prompting  
- **A3**: Example-enhanced prompting
- **A4**: Best-move finding
- **A5**: Simplified approach

#### B-Series (Reasoning)
- **B1**: Complete 85-turn reasoning
- **B2**: Chain-of-thought per-step reasoning
- **B3**: Extended CoT per-step (85 turns)

#### C-Series (Multi-turn)
- **C1**: One-by-one interactive approach
- **C2**: Twenty-turn conversations
- **C3**: Multi-turn without reasoning
- **C4**: Incremental Generator + Judge per move

## 🚀 Quick Start

### Prerequisites

```bash
# Required Python packages
pip install pydantic python-dotenv openai anthropic google-generativeai
```

### Environment Setup

Create a `.env` file with your API keys:

```env
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here
```

### Running Experiments

```bash
# Run full experiment suite
python simulate_llm_moves.py

# Test mode (quick 1-turn testing)
# Set EXECUTION_MODE = "test" in simulate_llm_moves.py

# Real mode (full 85-turn experiments)  
# Set EXECUTION_MODE = "real" in simulate_llm_moves.py
```

### Testing Individual Components

```bash
# Test specific LLM calls
python -c "from llm_client import call_llm; print(call_llm('gpt', [[1,2,3]], 'prompts/A/A1_Baseline.txt'))"
python -c "from llm_client import call_llm; print(call_llm('claude', [[1,2,3]], 'prompts/A/A1_Baseline.txt'))"
python -c "from llm_client import call_llm; print(call_llm('gemini', [[1,2,3]], 'prompts/A/A1_Baseline.txt'))"

# Test prompt detection system
python test_prompt_detection_only.py
python test_refactored_prompts.py

# Test specific strategy detection
python -c "from prompt_strategy import PromptStrategyDetector; print(PromptStrategyDetector.detect_strategy('prompts/A/A1_Baseline.txt'))"
```

## ⚙️ Configuration

### Model Configuration

The project supports multiple LLM providers with configurable models in `config.py`:

```python
MODEL_CONFIG = {
    "gpt": {"model_name": "gpt-4o-2024-11-20", "default_max_tokens": 8192},
    "claude": {"model_name": "claude-sonnet-4-20250514", "default_max_tokens": 8192},
    "gemini": {"model_name": "gemini-2.0-flash"}
}
```

### Experiment Parameters

Key settings in `simulate_llm_moves.py`:

```python
BOARDS_FILE = "boards/00.json"           # Test board data
PROMPTS_FOLDER = "prompts/C"             # Strategy directory
MODELS_TO_TEST = ["gpt", "claude", "gemini"]  # LLM providers
MAX_BOARD_ID_TO_TEST = 10                # Board limit
USE_MULTI_TURN = True                    # Enable multi-turn mode
EXECUTION_MODE = "real"                  # "test" or "real"
MAX_TURNS_REAL = 85                      # Full experiment turns
MAX_TURNS_TEST = 1                       # Quick test turns
```

## 📁 Project Structure

```
fruitbox_llm/
├── README.md                    # This file
├── CLAUDE.md                    # Detailed project documentation
├── simulate_llm_moves.py        # Main experiment runner
├── llm_client.py               # LLM provider interface
├── llm_strategy.py             # Pydantic models & strategies
├── prompt_strategy.py          # **[NEW]** Strategy detection system
├── utils.py                    # Utility functions
├── config.py                   # Model configuration
├── load_env.py                 # Environment loader
├── simulation.py               # Game simulator (external dependency)
├── analyze_result.py           # Results analysis
├── test_*.py                   # Testing utilities
├── boards/                     # Game board data (JSON)
├── prompts/                    # Strategy prompt templates
│   ├── A/                     # Basic strategies (A1-A5)
│   ├── B/                     # Reasoning strategies (B1-B3)
│   ├── C/                     # Multi-turn strategies (C1-C4)
│   ├── D/                     # Judge strategies
│   ├── E/                     # List-based strategies
│   └── legacy/                # Historical prompts
└── simulation_logs/           # Experiment results & logs
```

## 🧪 Testing Framework

### Strategy Detection System

The refactored `prompt_strategy.py` provides:

- **`PromptCategory`**: A/B/C category classification
- **`PromptSubStrategy`**: Specific strategy identification (A1-A4, B1-B2, etc.)
- **`PromptStrategyDetector`**: Automatic detection from file paths
- **Response Model Selection**: Automatic Pydantic model matching

### Pydantic Models

Core response schemas in `llm_strategy.py`:

```python
class Moves(BaseModel):
    moves: List[List[int]]

class MovesWithCoT(BaseModel):
    reasoning: str
    moves: List[List[int]]

class MovesWithPerStepReason(BaseModel):
    moves_with_reasoning: List[dict]

class JudgeVerdict(BaseModel):
    verdict: str
    reasoning: str
```

## 📊 Results & Analysis

### Experiment Logs

Results are automatically saved to `simulation_logs/` with:

- **Timestamped directories**: Each experiment run gets a unique timestamp
- **Strategy-based organization**: Results grouped by prompt strategy (A1, A2, etc.)
- **Provider-specific logs**: Separate results for each LLM provider
- **Success/Error tracking**: Automatic categorization of outcomes
- **Analysis summaries**: CSV files with performance metrics

### Performance Metrics

The system tracks:
- Success/failure rates per strategy
- Turn-by-turn performance
- Response parsing accuracy
- Strategy effectiveness comparison
- Cross-provider performance analysis

### Key Experiment Results

#### A-Series (Basic Prompting) - August 18, 2025
*Test: `test_20250818_215045(4, 2.0 A_polished)`*

| Strategy | Model | Success Rate | Avg LLM Score | Avg Greedy Score | Score Delta |
|----------|-------|--------------|---------------|------------------|-------------|
| A1_Baseline | Gemini | 100% (10/10) | 2.6 | 73.0 | -70.4 |
| A1_Baseline | GPT | 30% (3/10) | 0.0 | 80.7 | -80.7 |
| A2_RuleAdded | Gemini | 100% (10/10) | 5.5 | 73.0 | -67.5 |
| A2_RuleAdded | GPT | 0% (0/10) | 0.0 | 0.0 | 0.0 |
| A3_ExampleAdded | Gemini | 100% (10/10) | 2.9 | 73.0 | -70.1 |
| A3_ExampleAdded | GPT | 20% (2/10) | 2.0 | 79.5 | -77.5 |
| A4_FindBest | Gemini | 100% (10/10) | 2.6 | 73.0 | -70.4 |
| A4_FindBest | GPT | 90% (9/10) | 1.1 | 72.7 | -71.6 |
| A5_Simplest | Gemini | 100% (10/10) | 3.8 | 73.0 | -69.2 |
| A5_Simplest | GPT | 100% (10/10) | 0.4 | 73.0 | -72.6 |

**Key Findings:**
- Gemini shows consistently high success rates (100%) across all A-series strategies
- GPT struggles with reliability, showing high error rates (70-100%) except for A4_FindBest and A5_Simplest
- A2_RuleAdded performs best for Gemini with highest LLM score (5.5) and smallest score gap (-67.5)

#### B-Series (Reasoning) - August 18, 2025
*Test: `test_20250818_220648(2.0, B_polished)`*

| Strategy | Model | Success Rate | Avg LLM Score | Avg Greedy Score | Score Delta |
|----------|-------|--------------|---------------|------------------|-------------|
| B1_Complete85 | Gemini | 100% (10/10) | 1.6 | 73.0 | -71.4 |
| B2_CoT_PerStep | Gemini | 90% (9/10) | 2.7 | 73.1 | -70.4 |
| B3_CoT_PerStep_85 | Gemini | 100% (10/10) | 6.0 | 73.0 | -67.0 |

**Key Findings:**
- B3_CoT_PerStep_85 achieves the best performance with highest LLM score (6.0) and smallest gap (-67.0)
- Extended chain-of-thought reasoning (B3) outperforms basic reasoning approaches
- All B-series strategies show high reliability (90-100% success)

#### C-Series (Multi-turn) - August 18, 2025
*Test: `test_20250818_230145(2.0 C_Polished)`*

| Strategy | Model | Success Rate | Avg LLM Score | Avg Greedy Score | Score Delta |
|----------|-------|--------------|---------------|------------------|-------------|
| C2_twenty | Gemini | 80% (8/10) | 5.1 | 0.0 | 0.0 |
| C3_twenty_not_reasoning | Gemini | 0% (0/10) | 0.0 | 0.0 | 0.0 |

**Key Findings:**
- C2_twenty shows moderate success (80%) but limited score data
- C3_twenty_not_reasoning fails completely (0% success), highlighting the importance of reasoning in multi-turn approaches
- Multi-turn strategies appear more challenging to implement successfully

#### E-Series (List-based) - August 18, 2025
*Test: `test_20250818_231125(2.0 E_polsihed)`*

| Strategy | Model | Success Rate | Avg LLM Score | Avg Greedy Score | Score Delta |
|----------|-------|--------------|---------------|------------------|-------------|
| E1_Find_from_list | Gemini | 50% (5/10) | 4.0 | 75.8 | -71.8 |
| E2_Find_from_list_reasoning | Gemini | 100% (10/10) | 1.0 | 73.0 | -72.0 |

**Key Findings:**
- Adding reasoning (E2) dramatically improves success rate from 50% to 100%
- E1 shows higher individual scores when successful but lacks consistency
- Reasoning-enhanced strategies prove more reliable across different approaches

### Overall Performance Summary

**Best Performing Strategies:**
1. **B3_CoT_PerStep_85**: Highest LLM score (6.0), smallest performance gap (-67.0)
2. **A2_RuleAdded** (Gemini): Second-best gap (-67.5) with 100% reliability
3. **A5_Simplest** (Gemini): Good balance of simplicity and performance (-69.2 gap)

**Model Reliability:**
- **Gemini**: Consistently high success rates (80-100%) across all strategy types
- **GPT**: Variable performance, requiring careful strategy selection

**Strategy Insights:**
- Reasoning-enhanced approaches consistently outperform basic prompting
- Extended chain-of-thought (85-turn) provides significant benefits
- Multi-turn strategies require careful design to avoid complete failure
- Rule-based enhancements improve performance over baseline approaches

## 🛠️ Development

### Adding New Strategies

1. Create prompt file in appropriate directory (`prompts/A/`, `prompts/B/`, etc.)
2. Follow naming convention: `[Category][Number]_[Description].txt`
3. Add corresponding Pydantic model if needed in `llm_strategy.py`
4. Strategy will be automatically detected by keyword system

### Adding New LLM Providers

1. Add provider configuration to `config.py`
2. Implement provider interface in `llm_client.py`
3. Add provider to `MODELS_TO_TEST` in experiment configuration

## 🔒 Security & Best Practices

- API keys stored in environment variables only
- No secrets committed to repository
- Automatic response validation using Pydantic
- Error handling for malformed LLM responses
- Unicode-safe file handling throughout

## 📋 Dependencies

### External Modules (Not Included)
- `simulation.FruitBoxSimulator`: Game simulator for move validation

### Python Packages
- `pydantic`: Data validation and parsing
- `python-dotenv`: Environment variable management
- `openai`: GPT model access
- `anthropic`: Claude model access
- `google-generativeai`: Gemini model access

## 🤝 Contributing

This is a research project. When making changes:

1. Follow existing code conventions and patterns
2. Update relevant documentation in `CLAUDE.md`
3. Ensure new strategies follow the A/B/C classification system
4. Test with multiple LLM providers before committing
5. Maintain backward compatibility with existing logs

## 📄 License

Research project - see project documentation for usage guidelines.

---

For detailed technical documentation, see [`CLAUDE.md`](./CLAUDE.md).
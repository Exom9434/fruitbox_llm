# prompt_strategy.py
"""
Prompt handling system using keyword-based strategy detection (A, B, C)
"""

import re
from typing import Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from llm_strategy import (
    Moves, SimpleMoves, MovesWithCoT, MovesWithPerStepReason, 
    SingleMoveWrapper, SingleMoveWithReasonWrapper, JudgeVerdict
)


class PromptCategory(Enum):
    """Main prompt categories"""
    A = "A"  # Basic strategies
    B = "B"  # Reasoning strategies  
    C = "C"  # Multi-turn incremental strategies
    D = "D"  # Multi-turn judge strategies
    E = "E"  # List-based strategies


class PromptSubStrategy(Enum):
    """Sub-strategies within each category"""
    # A-Series (Basic)
    A1 = "A1"  # Baseline
    A2 = "A2"  # Rule Added
    A3 = "A3"  # Example Added
    A4 = "A4"  # Find Best
    A5 = "A5"  # Simplest
    
    # B-Series (Reasoning)
    B1 = "B1"  # Chain of Thought
    B2 = "B2"  # Per-Step Reasoning
    B3 = "B3"  # CoT Per-Step 85 moves
    
    # C-Series (Multi-turn incremental)
    C1 = "C1"  # Incremental one move at a time
    C2 = "C2"  # Batch 20 moves x 4 iterations
    C3 = "C3"  # Multi-turn 20 moves without reasoning
    
    # D-Series (Multi-turn judge)
    D1 = "D1"  # Generator + Judge full evaluation
    
    # E-Series (List-based)
    E1 = "E1"  # Find from list
    E2 = "E2"  # Find from list with reasoning


@dataclass
class PromptStrategy:
    """Configuration for a specific prompt strategy"""
    category: PromptCategory
    sub_strategy: PromptSubStrategy
    strategy_name: str
    response_model: Optional[Any]
    is_multi_turn: bool
    requires_generator: bool = False
    requires_judge: bool = False


class PromptStrategyDetector:
    """Detects and configures prompt strategies based on keywords A, B, C"""
    
    # Strategy configurations
    STRATEGIES = {
        PromptSubStrategy.A1: PromptStrategy(
            category=PromptCategory.A,
            sub_strategy=PromptSubStrategy.A1,
            strategy_name="A-1_Baseline",
            response_model=SimpleMoves,
            is_multi_turn=False
        ),
        PromptSubStrategy.A2: PromptStrategy(
            category=PromptCategory.A,
            sub_strategy=PromptSubStrategy.A2,
            strategy_name="A-2_RuleAdded",
            response_model=SimpleMoves,
            is_multi_turn=False
        ),
        PromptSubStrategy.A3: PromptStrategy(
            category=PromptCategory.A,
            sub_strategy=PromptSubStrategy.A3,
            strategy_name="A-3_ExampleAdded",
            response_model=SimpleMoves,
            is_multi_turn=False
        ),
        PromptSubStrategy.A4: PromptStrategy(
            category=PromptCategory.A,
            sub_strategy=PromptSubStrategy.A4,
            strategy_name="A-4_FindBest",
            response_model=SimpleMoves,
            is_multi_turn=False
        ),
        PromptSubStrategy.A5: PromptStrategy(
            category=PromptCategory.A,
            sub_strategy=PromptSubStrategy.A5,
            strategy_name="A-5_Simplest",
            response_model=SimpleMoves,
            is_multi_turn=False
        ),
        PromptSubStrategy.B1: PromptStrategy(
            category=PromptCategory.B,
            sub_strategy=PromptSubStrategy.B1,
            strategy_name="B-1_CoT",
            response_model=MovesWithCoT,
            is_multi_turn=False
        ),
        PromptSubStrategy.B2: PromptStrategy(
            category=PromptCategory.B,
            sub_strategy=PromptSubStrategy.B2,
            strategy_name="B-2_PerStepReason",
            response_model=MovesWithPerStepReason,
            is_multi_turn=False
        ),
        PromptSubStrategy.B3: PromptStrategy(
            category=PromptCategory.B,
            sub_strategy=PromptSubStrategy.B3,
            strategy_name="B-3_CoT_PerStep_85",
            response_model=MovesWithPerStepReason,
            is_multi_turn=False
        ),
        PromptSubStrategy.C1: PromptStrategy(
            category=PromptCategory.C,
            sub_strategy=PromptSubStrategy.C1,
            strategy_name="C-1_Incremental_One_Move",
            response_model=SingleMoveWrapper,
            is_multi_turn=True
        ),
        PromptSubStrategy.C2: PromptStrategy(
            category=PromptCategory.C,
            sub_strategy=PromptSubStrategy.C2,
            strategy_name="C-2_Batch_20_Moves",
            response_model=MovesWithPerStepReason,
            is_multi_turn=True
        ),
        PromptSubStrategy.C3: PromptStrategy(
            category=PromptCategory.C,
            sub_strategy=PromptSubStrategy.C3,
            strategy_name="C-3_Twenty_Not_Reasoning",
            response_model=SimpleMoves,
            is_multi_turn=True
        ),
        PromptSubStrategy.D1: PromptStrategy(
            category=PromptCategory.D,
            sub_strategy=PromptSubStrategy.D1,
            strategy_name="D-1_Judge_Session",
            response_model=Moves,
            is_multi_turn=True,
            requires_generator=True,
            requires_judge=True
        ),
        PromptSubStrategy.E1: PromptStrategy(
            category=PromptCategory.E,
            sub_strategy=PromptSubStrategy.E1,
            strategy_name="E-1_Find_from_list",
            response_model=SimpleMoves,
            is_multi_turn=False
        ),
        PromptSubStrategy.E2: PromptStrategy(
            category=PromptCategory.E,
            sub_strategy=PromptSubStrategy.E2,
            strategy_name="E-2_Find_from_list_reasoning",
            response_model=MovesWithPerStepReason,
            is_multi_turn=False
        ),
    }

    @classmethod
    def detect_strategy(cls, prompt_path: str) -> Optional[PromptStrategy]:
        """
        Detect strategy from prompt path using keywords A, B, C
        
        Args:
            prompt_path: Path to the prompt file
            
        Returns:
            PromptStrategy configuration or None if not detected
        """
        prompt_name = prompt_path.lower()
        
        # Extract strategy pattern (A1, A2, B1, B2, C1, C3, D1, E1, etc.)
        pattern = r'[abcde][1-9]'
        matches = re.findall(pattern, prompt_name)
        
        if not matches:
            return None
        
        # Use the first match found
        strategy_key = matches[0].upper()
        
        try:
            sub_strategy = PromptSubStrategy(strategy_key)
            return cls.STRATEGIES.get(sub_strategy)
        except ValueError:
            return None

    @classmethod
    def get_strategy_by_category(cls, category: PromptCategory) -> list[PromptStrategy]:
        """Get all strategies for a specific category"""
        return [strategy for strategy in cls.STRATEGIES.values() 
                if strategy.category == category]

    @classmethod
    def is_category(cls, prompt_path: str, category: PromptCategory) -> bool:
        """Check if prompt belongs to a specific category"""
        strategy = cls.detect_strategy(prompt_path)
        return strategy is not None and strategy.category == category

    @classmethod
    def is_multi_turn_strategy(cls, prompt_path: str) -> bool:
        """Check if strategy requires multi-turn processing"""
        strategy = cls.detect_strategy(prompt_path)
        return strategy is not None and strategy.is_multi_turn

    @classmethod
    def requires_judge(cls, prompt_path: str) -> bool:
        """Check if strategy requires judge functionality"""
        strategy = cls.detect_strategy(prompt_path)
        return strategy is not None and strategy.requires_judge

    @classmethod
    def requires_generator(cls, prompt_path: str) -> bool:
        """Check if strategy requires generator functionality"""
        strategy = cls.detect_strategy(prompt_path)
        return strategy is not None and strategy.requires_generator


# Convenience functions for backward compatibility
def is_a_series(prompt_path: str) -> bool:
    """Check if prompt is A-series (Basic strategies)"""
    return PromptStrategyDetector.is_category(prompt_path, PromptCategory.A)

def is_b_series(prompt_path: str) -> bool:
    """Check if prompt is B-series (Reasoning strategies)"""
    return PromptStrategyDetector.is_category(prompt_path, PromptCategory.B)

def is_c_series(prompt_path: str) -> bool:
    """Check if prompt is C-series (Multi-turn incremental strategies)"""
    return PromptStrategyDetector.is_category(prompt_path, PromptCategory.C)

def is_d_series(prompt_path: str) -> bool:
    """Check if prompt is D-series (Multi-turn judge strategies)"""
    return PromptStrategyDetector.is_category(prompt_path, PromptCategory.D)

def is_e_series(prompt_path: str) -> bool:
    """Check if prompt is E-series (List-based strategies)"""
    return PromptStrategyDetector.is_category(prompt_path, PromptCategory.E)

def get_response_model_for_prompt(prompt_path: str) -> Optional[Any]:
    """Get the appropriate Pydantic response model for a prompt"""
    strategy = PromptStrategyDetector.detect_strategy(prompt_path)
    return strategy.response_model if strategy else Moves

def get_strategy_name(prompt_path: str) -> str:
    """Get human-readable strategy name"""
    strategy = PromptStrategyDetector.detect_strategy(prompt_path)
    return strategy.strategy_name if strategy else "Unknown_Strategy"
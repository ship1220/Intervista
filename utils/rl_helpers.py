"""
Reinforcement Learning Helpers (Simplified for Contextual Bandit)
==================================================================

Essential utilities for bandit-based adaptive learning:
- State discretization (convert continuous metrics to discrete states)
- Reward calculation (simple score improvement signal)

Features:
- No temporal difference (no next_state dependency)
- Running average of rewards per (state, action)
- Simple, interpretable reward signal
"""

import logging

logger = logging.getLogger(__name__)


# =============================================================================
# STATE DISCRETIZATION
# =============================================================================

def get_state_id(avg_score: float, weak_topics_count: int) -> str:
    """
    Convert continuous user metrics to a discrete state.
    
    Args:
        avg_score: User's average score (0-100)
        weak_topics_count: Number of weak topics identified
    
    Returns:
        State ID in format "{score_level}-{weak_count}"
        Examples: "low-2", "medium-3", "high-0"
    
    Discretization:
        - avg_score < 50 → "low"
        - 50-75 → "medium"
        - > 75 → "high"
        - weak_count: exact count (0, 1, 2, 3+)
    """
    # Normalize score to 0-100 range
    avg_score = max(0, min(100, float(avg_score)))
    
    # Discretize score level
    if avg_score < 50:
        score_level = "low"
    elif avg_score <= 75:
        score_level = "medium"
    else:
        score_level = "high"
    
    # Cap weak topics count at 3+ 
    weak_count = min(int(weak_topics_count), 3)
    
    state_id = f"{score_level}-{weak_count}"
    
    logger.debug(
        f"State: score={avg_score:.1f} weak_topics={weak_topics_count} → state_id={state_id}"
    )
    
    return state_id


# =============================================================================
# REWARD CALCULATION (SIMPLE SCORE IMPROVEMENT)
# =============================================================================

def calculate_reward(current_score: float, previous_score: float) -> float:
    """
    Calculate simple reward based on score improvement.
    
    Args:
        current_score: User's current score (0-100)
        previous_score: User's previous score (0-100)
    
    Returns:
        Reward normalized to [-1.0, +1.0]
    
    Formula:
        improvement = current_score - previous_score
        reward = improvement / 100
        reward = clamp(reward, -1, +1)
    
    Interpretation:
        - +1.0: Improved 100+ points (capped at +1)
        - +0.15: Improved 15 points
        - 0.0: No change
        - -0.30: Declined 30 points
        - -1.0: Declined 100+ points (capped at -1)
    """
    # Normalize scores to [0, 100]
    curr = max(0.0, min(100.0, float(current_score)))
    prev = max(0.0, min(100.0, float(previous_score)))
    
    # Calculate improvement
    improvement = curr - prev
    
    # Normalize to [-1, +1]
    reward = improvement / 100.0
    reward = max(-1.0, min(1.0, reward))
    
    logger.debug(
        f"Reward: current={curr:.1f}, previous={prev:.1f}, "
        f"improvement={improvement:.1f} → reward={reward:.3f}"
    )
    
    return reward


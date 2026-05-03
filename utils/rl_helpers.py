"""
Reinforcement Learning Helpers
===============================

Utilities for:
- State discretization (convert continuous metrics to discrete states)
- Reward calculation (compute reward from metrics and improvement)
- Q-table updates and value retrieval
- Logging of RL metrics for debugging

All functions are ADDITIVE and do not modify existing interview/course logic.
"""

import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# STATE DISCRETIZATION
# =============================================================================

def get_state_id(
    avg_score: float,
    weak_topics_count: int,
    session_count: int = 0
) -> str:
    """
    Convert continuous user metrics to a discrete state.
    
    Args:
        avg_score: User's average score (0-100)
        weak_topics_count: Number of weak topics identified
        session_count: Number of sessions completed (unused in current discretization)
    
    Returns:
        State ID in format "{score_level}_{weak_flag}"
        Examples: "low_weak", "medium_strong", "high_weak"
    
    Rule:
        - avg_score < 50 → "low"
        - 50-75 → "medium"
        - > 75 → "high"
        - weak_topics_count > 0 → "weak"
        - weak_topics_count == 0 → "strong"
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
    
    # Discretize weak topics
    weak_flag = "weak" if weak_topics_count > 0 else "strong"
    
    state_id = f"{score_level}_{weak_flag}"
    
    logger.debug(
        f"state_discretized: avg_score={avg_score:.1f} weak_topics={weak_topics_count} "
        f"→ state_id={state_id}"
    )
    
    return state_id


# =============================================================================
# REWARD CALCULATION
# =============================================================================

def calculate_reward(
    metrics: Dict[str, float],
    previous_score: float,
    current_score: float
) -> float:
    """
    Calculate reward based on evaluation metrics and improvement.
    
    Args:
        metrics: Dict with keys C, K, F, S (all normalized 0.0-1.0)
            - C: Correctness
            - K: Knowledge/Confidence
            - F: Fluency
            - S: Sentiment
        previous_score: User's previous assessment score (0-100)
        current_score: User's current assessment score (0-100)
    
    Returns:
        Reward value normalized to 0.0-1.0
    
    Formula:
        R = 0.3*C + 0.2*K + 0.2*F + 0.1*S + 0.2*Δ
        where Δ = max(current_score - previous_score, 0) / 100
    
    This rewards both absolute quality (C, K, F, S) and improvement (Δ).
    """
    # Validate metrics
    C = max(0.0, min(1.0, float(metrics.get("C", 0.0))))
    K = max(0.0, min(1.0, float(metrics.get("K", 0.0))))
    F = max(0.0, min(1.0, float(metrics.get("F", 0.0))))
    S = max(0.0, min(1.0, float(metrics.get("S", 0.0))))
    
    # Normalize improvement (delta) term
    prev_score = max(0, min(100, float(previous_score)))
    curr_score = max(0, min(100, float(current_score)))
    improvement = max(curr_score - prev_score, 0)
    delta = improvement / 100.0  # Normalize to 0-1
    
    # Compute weighted reward
    reward = (0.3 * C) + (0.2 * K) + (0.2 * F) + (0.1 * S) + (0.2 * delta)
    reward = max(0.0, min(1.0, reward))  # Clamp to 0-1
    
    logger.debug(
        f"reward_calculated: C={C:.2f} K={K:.2f} F={F:.2f} S={S:.2f} "
        f"delta={delta:.2f} → reward={reward:.3f}"
    )
    
    return reward


def normalize_final_score(raw_score: float) -> float:
    """
    Normalize any raw score to 0-100 range.
    
    Args:
        raw_score: Raw score from LLM or calculation
    
    Returns:
        Score clamped to [0, 100]
    """
    try:
        normalized = float(raw_score)
    except (TypeError, ValueError):
        normalized = 0.0
    
    return max(0.0, min(100.0, normalized))


# =============================================================================
# Q-TABLE OPERATIONS
# =============================================================================

def compute_q_value_update(
    current_q: float,
    reward: float,
    next_state_max_q: float,
    alpha: float = 0.1,
    gamma: float = 0.99
) -> float:
    """
    Compute updated Q-value using standard RL update rule.
    
    Args:
        current_q: Current Q(s, a)
        reward: Reward received for taking action a in state s
        next_state_max_q: max_a' Q(s', a')
        alpha: Learning rate (default 0.1)
        gamma: Discount factor (default 0.99)
    
    Returns:
        Updated Q-value
    
    Formula:
        Q_new = Q_old + alpha * (R + gamma * max_Q(s') - Q_old)
    """
    alpha = max(0.0, min(1.0, float(alpha)))
    gamma = max(0.0, min(1.0, float(gamma)))
    
    current_q = float(current_q)
    reward = float(reward)
    next_state_max_q = float(next_state_max_q)
    
    td_error = reward + gamma * next_state_max_q - current_q
    q_new = current_q + alpha * td_error
    
    logger.debug(
        f"q_value_updated: old_q={current_q:.3f} reward={reward:.3f} "
        f"next_max_q={next_state_max_q:.3f} → new_q={q_new:.3f} (td_error={td_error:.3f})"
    )
    
    return q_new


# =============================================================================
# METRICS VALIDATION
# =============================================================================

def validate_rl_metrics(metrics: Dict[str, float]) -> Tuple[bool, str]:
    """
    Validate that metrics dict contains required keys with valid values.
    
    Args:
        metrics: Dict expected to have C, K, F, S keys
    
    Returns:
        Tuple of (is_valid, error_message)
        - If valid: (True, "")
        - If invalid: (False, error_message)
    """
    required_keys = {"C", "K", "F", "S"}
    
    if not isinstance(metrics, dict):
        return False, "Metrics must be a dictionary"
    
    if not required_keys.issubset(metrics.keys()):
        missing = required_keys - set(metrics.keys())
        return False, f"Missing required metric keys: {missing}"
    
    for key in required_keys:
        try:
            val = float(metrics[key])
            if not (0.0 <= val <= 1.0):
                return False, f"Metric {key}={val} out of range [0.0, 1.0]"
        except (TypeError, ValueError):
            return False, f"Metric {key} is not a valid float"
    
    return True, ""


# =============================================================================
# LOGGING HELPERS
# =============================================================================

def log_rl_metrics(
    user_id: int,
    state_id: str,
    metrics: Dict[str, float],
    reward: float,
    context: Optional[str] = None
) -> None:
    """
    Log RL metrics for debugging and analysis.
    
    Args:
        user_id: User ID
        state_id: Current state ID
        metrics: Evaluation metrics (C, K, F, S)
        reward: Computed reward
        context: Optional context string (e.g., "interview", "quiz")
    """
    context_str = f" [{context}]" if context else ""
    logger.info(
        f"RL_METRICS{context_str}: user_id={user_id} state_id={state_id} "
        f"C={metrics.get('C', 0):.2f} K={metrics.get('K', 0):.2f} "
        f"F={metrics.get('F', 0):.2f} S={metrics.get('S', 0):.2f} "
        f"reward={reward:.3f}"
    )

"""
Reinforcement Learning Service
================================

Implements tabular Q-learning for adaptive interview and course selection.

Key Features:
- ε-greedy action selection with cold-start handling
- Adaptive learning rate (decay with visit count)
- State-action Q-value management
- Production-safe logging

Usage:
    learner = TabularQLearner(db_session)
    action = learner.select_action(state_id, user_state)
    learner.update_q_table(state_id, action_id, reward, next_state_id)
"""

import logging
import random
from typing import Optional, Tuple
from sqlalchemy.orm import Session
from models import QTable, UserState

logger = logging.getLogger(__name__)

# ============================================================================
# ACTION SPACE DEFINITIONS
# ============================================================================

INTERVIEW_ACTIONS = {
    "ask_easy_question": 0,
    "ask_medium_question": 1,
    "ask_hard_question": 2,
    "ask_resume_question": 3,
    "ask_behavioral_question": 4,
}

COURSE_ACTIONS = {
    "revision": 0,
    "easy": 1,
    "mixed": 2,
    "advanced": 3,
}

# Hyperparameters
GAMMA = 0.9  # Discount factor
EPSILON = 0.2  # Exploration probability
COLD_START_THRESHOLD = 3  # Sessions before using learned policy


class TabularQLearner:
    """
    Tabular Q-Learning agent for adaptive interview and course selection.

    This learner maintains a Q-table of state-action values and uses
    ε-greedy exploration to balance exploration vs exploitation.
    """

    def __init__(self, db: Session, action_space: str = "interview"):
        """
        Initialize the Q-learner.

        Args:
            db: SQLAlchemy session for database access
            action_space: Either 'interview' or 'course'
        """
        self.db = db
        self.action_space = action_space

        if action_space == "interview":
            self.actions = INTERVIEW_ACTIONS
        elif action_space == "course":
            self.actions = COURSE_ACTIONS
        else:
            raise ValueError(f"Unknown action space: {action_space}")

        logger.info(f"QLearner initialized for {action_space} with {len(self.actions)} actions")

    # ========================================================================
    # ACTION SELECTION (ε-GREEDY)
    # ========================================================================

    def select_action(self, state_id: str, user_state: Optional[UserState]) -> str:
        """
        Select an action using ε-greedy strategy with cold-start handling.

        Cold-start (first 3 sessions):
        - Force easy actions to bootstrap learning
        - Interview: ask_easy_question
        - Course: teach_easy

        After cold-start:
        - 80% probability: select action with highest Q-value
        - 20% probability: select random action

        Args:
            state_id: Current discretized state
            user_state: User state record (optional, for session count)

        Returns:
            Selected action name (string)
        """
        session_count = user_state.session_count if user_state else 0

        # ====== COLD START ======
        if session_count < COLD_START_THRESHOLD:
            action = self._cold_start_action()
            logger.info(
                f"COLD_START: user_session_count={session_count} state_id={state_id} → action={action}"
            )
            return action

        # ====== EXPLOIT vs EXPLORE ======
        if random.random() < (1 - EPSILON):
            # EXPLOIT: choose greedy action
            action = self._greedy_action(state_id)
            logger.debug(
                f"EXPLOIT: state_id={state_id} → action={action} (ε-greedy, greedy branch)"
            )
        else:
            # EXPLORE: choose random action
            action = random.choice(list(self.actions.keys()))
            logger.debug(
                f"EXPLORE: state_id={state_id} → action={action} (ε-greedy, explore branch)"
            )

        return action

    def _cold_start_action(self) -> str:
        """Return the cold-start action for this action space."""
        if self.action_space == "interview":
            return "ask_easy_question"
        else:
            return "easy"

    def _greedy_action(self, state_id: str) -> str:
        """
        Select the action with the highest Q-value for a given state.

        If no Q-values exist for the state, return a random action
        and initialize Q-values to 0.

        Args:
            state_id: Current state ID

        Returns:
            Action with highest Q-value
        """
        best_action = None
        best_q = float("-inf")

        for action_name, action_id in self.actions.items():
            q_value = self._get_q_value(state_id, action_name)

            if q_value > best_q:
                best_q = q_value
                best_action = action_name

        if best_action is None:
            best_action = random.choice(list(self.actions.keys()))
            logger.debug(f"GREEDY: No Q-values for state_id={state_id}, defaulting to random")

        return best_action

    # ========================================================================
    # Q-TABLE OPERATIONS
    # ========================================================================

    def _get_q_value(self, state_id: str, action_name: str) -> float:
        """
        Fetch Q-value for (state, action) pair.

        If not found in database, returns 0.0 (optimistic initialization).

        Args:
            state_id: State identifier
            action_name: Action name

        Returns:
            Q-value (float), or 0.0 if not found
        """
        try:
            q_record = (
                self.db.query(QTable)
                .filter(
                    QTable.state_id == state_id,
                    QTable.action_id == action_name,
                )
                .first()
            )

            if q_record:
                return float(q_record.q_value)
            else:
                return 0.0
        except Exception as e:
            logger.error(f"Error fetching Q-value: {e}")
            return 0.0

    def _get_max_q_next_state(self, next_state_id: str) -> float:
        """
        Get the maximum Q-value for all actions in the next state.

        Used for the temporal difference update:
        Q(s,a) ← Q(s,a) + α [ R + γ * max Q(s',a') - Q(s,a) ]

        Args:
            next_state_id: Next state identifier

        Returns:
            Maximum Q-value in next state, or 0.0 if no actions exist
        """
        try:
            max_q = 0.0
            for action_name in self.actions.keys():
                q = self._get_q_value(next_state_id, action_name)
                max_q = max(max_q, q)
            return max_q
        except Exception as e:
            logger.error(f"Error fetching max Q for next state: {e}")
            return 0.0

    # ========================================================================
    # Q-TABLE UPDATE (MAIN LEARNING STEP)
    # ========================================================================

    def update_q_table(
        self,
        state_id: str,
        action_id: str,
        reward: float,
        next_state_id: str,
    ) -> Tuple[float, float]:
        """
        Update Q-value using temporal difference learning.

        Formula:
            α = 1 / (1 + visit_count)
            Q(s,a) ← Q(s,a) + α [ R + γ * max Q(s',a') - Q(s,a) ]

        Args:
            state_id: Current state
            action_id: Selected action
            reward: Received reward (normalized 0-1)
            next_state_id: Next state

        Returns:
            Tuple of (old_q_value, new_q_value)
        """
        try:
            # Fetch or create Q-record
            q_record = (
                self.db.query(QTable)
                .filter(
                    QTable.state_id == state_id,
                    QTable.action_id == action_id,
                )
                .first()
            )

            if q_record is None:
                q_record = QTable(
                    state_id=state_id,
                    action_id=action_id,
                    q_value=0.0,
                    visit_count=0,
                )
                self.db.add(q_record)
                self.db.flush()

            old_q = float(q_record.q_value)

            # Adaptive learning rate: α = 1 / (1 + visit_count)
            alpha = 1.0 / (1.0 + q_record.visit_count)

            # Temporal difference: R + γ * max Q(s',a') - Q(s,a)
            max_q_next = self._get_max_q_next_state(next_state_id)
            td_error = reward + GAMMA * max_q_next - old_q

            # Update Q-value
            new_q = old_q + alpha * td_error
            q_record.q_value = new_q
            q_record.visit_count += 1

            self.db.commit()

            logger.info(
                f"Q_UPDATE: state_id={state_id} action_id={action_id} "
                f"reward={reward:.3f} α={alpha:.4f} "
                f"Q_old={old_q:.3f} → Q_new={new_q:.3f} "
                f"td_error={td_error:.3f} visit_count={q_record.visit_count}"
            )

            return old_q, new_q

        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating Q-table: {e}", exc_info=True)
            return 0.0, 0.0

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def get_q_value_dict(self, state_id: str) -> dict:
        """
        Get all Q-values for a given state.

        Useful for logging and debugging.

        Args:
            state_id: State identifier

        Returns:
            Dict mapping action names to Q-values
        """
        try:
            q_dict = {}
            for action_name in self.actions.keys():
                q_dict[action_name] = self._get_q_value(state_id, action_name)
            return q_dict
        except Exception as e:
            logger.error(f"Error fetching Q-value dict: {e}")
            return {}

    def reset_q_table(self) -> None:
        """
        DANGER: Reset all Q-values to 0.

        Use only for testing or policy reset. NOT recommended in production.
        """
        try:
            self.db.query(QTable).delete()
            self.db.commit()
            logger.warning("Q-TABLE RESET: All Q-values deleted")
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error resetting Q-table: {e}")

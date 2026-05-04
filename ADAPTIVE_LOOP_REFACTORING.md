# Adaptive Loop Refactoring: Bandit-Based Course Generation

**Status**: ✅ COMPLETE
**Date**: May 4, 2026
**Scope**: `/api/interview/evaluate` endpoint refactored for bandit-driven adaptive learning

---

## 🎯 Objectives Achieved

✅ Bandit action selection drives course generation
✅ Strict action-to-topics-difficulty mapping (non-negotiable)
✅ Course generation ALWAYS runs (with fallback)
✅ Simple reward = score improvement (not metric-based)
✅ Explicit topics and difficulty passed to LLM
✅ Structured logging for complete visibility
✅ API response format UNCHANGED (backward compatible)

---

## 📊 Adaptive Loop Flow

```
Interview Completion
    ↓
Score Computation (overall, content_avg, speech metrics)
    ↓
State Computation: state_id = get_state_id(overall, weak_count)
    ↓
Bandit Action Selection: action = bandit.select_action(state_id, user_state)
    ↓
STRICT MAPPING: action → (topics, difficulty)
    ├─ "revision"  → (fundamentals, easy)
    ├─ "easy"      → (fundamentals, easy)
    ├─ "mixed"     → (weak + related, medium)
    └─ "advanced"  → (advanced topics, hard)
    ↓
Course Generation: create_course_internal(..., topics=X, difficulty=Y)
    ↓
Reward Calculation: reward = (overall - previous_avg) / 100, normalized [-1, 1]
    ↓
Bandit Update: bandit.update_action_value(prev_state, action, reward)
    ↓
UserState Update: Track session, state, weak_topics, avg_score
    ↓
Response: {new_course, recommended_action, rl_metrics}
```

---

## 🔄 Modified Components

### 1. `create_course_internal()` Function Signature

**Before:**
```python
async def create_course_internal(
    user, role, level, weak_topics, action, db
) → int | None
```

**After:**
```python
async def create_course_internal(
    user, role, level, weak_topics, action, db,
    topics: list[str] = None,        # EXPLICIT topics for course
    difficulty: str = "medium"       # EXPLICIT difficulty level
) → int | None
```

**Changes:**
- Added `topics` parameter: LLM receives explicit topic list (not computed internally)
- Added `difficulty` parameter: LLM enforces specific difficulty level
- Prompt updated to reference both parameters
- Difficulty level embedded in module generation instructions
- Course title now includes action: `f"{level} {role} - {action.title()} Course"`

### 2. `/api/interview/evaluate` - RL Section (10-Step Process)

**STEP 1: State Computation**
```python
state_id = get_state_id(overall, len(weak_topics))
# Returns: "low-1", "medium-2", "high-3" (discretized state)
```

**STEP 2: Retrieve Previous Score**
```python
previous_score = user_state.avg_score if exists else overall
# Use average score (not last_score) as baseline
```

**STEP 3: Bandit Action Selection**
```python
action = course_learner.select_action(state_id, user_state)
# Returns: "revision", "easy", "mixed", or "advanced"
# Cold-start: Returns "easy" for first 2 sessions
# Then: 80% exploit best Q-value, 20% explore random
```

**STEP 4: Strict Action-to-Mapping**
```python
if action == "revision":
    topics = get_fundamentals(weak_topics)          # First 2 weak topics
    difficulty = "easy"
elif action == "easy":
    topics = get_fundamentals(weak_topics)          # First 2 weak topics
    difficulty = "easy"
elif action == "mixed":
    topics = get_related_topics(weak_topics)        # Weak + expanded
    difficulty = "medium"
elif action == "advanced":
    topics = get_advanced_topics(weak_topics)       # Advanced variants
    difficulty = "hard"
```

**Topic Generation Helpers:**
- `get_fundamentals(topics)`: Return first 2 weak topics (core learning)
- `get_related_topics(topics)`: Weak topics + logical expansions (balanced learning)
- `get_advanced_topics(topics)`: "advanced {topic}" variants + system design (deep learning)

**STEP 5: Course Generation (ALWAYS)**
```python
new_course_id = await create_course_internal(
    user, role, level, weak_topics, action, db,
    topics=course_topics,          # EXPLICIT
    difficulty=course_difficulty   # EXPLICIT
)
# If fails: Fallback to revision action with fundamentals + easy
# If still fails: new_course_id = None (graceful degradation)
```

**STEP 6: Fetch Course Details**
```python
if new_course_id:
    new_course = {
        "course_id": new_course_id,
        "title": course_title
    }
```

**STEP 7: Calculate Reward**
```python
score_improvement = overall - previous_score         # Range: -100 to +100
reward = score_improvement / 100.0                  # Normalize: [-1, +1]
reward = max(min(reward, 1.0), -1.0)                # Clamp: [-1, +1]
```

**Key Insight:** Reward is IMMEDIATE and SIMPLE
- Positive reward when student improves
- Negative reward when performance drops
- Bandit learns which action leads to improvement in each state

**STEP 8: Update Bandit Q-Value**
```python
bandit.update_action_value(prev_state_id, action, reward)
# Running average: new_q = (old_q * count + reward) / (count + 1)
# No next_state dependency - immediate feedback loop
```

**STEP 9: Update UserState**
```python
user_state.state_id = state_id                      # Current state
user_state.last_score = overall                     # Last interview score
user_state.avg_score = rolling_average              # Moving average
user_state.current_proficiency = avg_score / 100    # Proficiency level
user_state.weak_topics = weak_topics                # Current weaknesses
user_state.session_count += 1                       # Session counter
```

**STEP 10: Populate Report Response**
```python
report["new_course_id"] = new_course_id
report["new_course"] = new_course
report["recommended_action"] = action
report["rl_metrics"] = {
    "state": state_id,
    "previous_state": prev_state_id,
    "action": action,
    "course_topics": course_topics,
    "course_difficulty": course_difficulty,
    "reward": reward,
    "score_improvement": score_improvement,
    "new_course_id": new_course_id,
}
```

---

## 📋 API Response Format (UNCHANGED)

**Response Structure:**
```json
{
  "candidate_profile": {...},
  "overall_score": 75,
  "verdict": "Strong candidate",
  "performance_summary": "...",
  "voice_analysis": {...},
  "content_analysis": {...},
  "detailed_answers": [...],
  "weak_topics": [...],
  
  // ✅ NEW/UPDATED FIELDS (backward compatible)
  "new_course": {
    "course_id": 42,
    "title": "mid Software Developer - mixed Course"
  },
  "recommended_action": "mixed",
  "rl_metrics": {
    "state": "medium-2",
    "previous_state": "medium-1",
    "action": "mixed",
    "course_topics": ["weak_topic_1", "weak_topic_2", "..."],
    "course_difficulty": "medium",
    "reward": 0.15,
    "score_improvement": 15,
    "new_course_id": 42
  }
}
```

---

## 🔐 Backward Compatibility

✅ **API Response**: All existing fields preserved
✅ **Database Schema**: UserState and Course tables unchanged
✅ **Fallback Logic**: Course generation always attempts
✅ **Error Handling**: Graceful degradation if course generation fails

---

## 🎓 Action-to-Learning-Path Mapping

| Action | Topics | Difficulty | Best For | Learning Path |
|--------|--------|-----------|----------|---------------|
| **revision** | Weak (first 2) | Easy | First session, low performers | Deep practice on core gaps |
| **easy** | Weak (first 2) | Easy | Struggling learners | Confidence building |
| **mixed** | Weak + related | Medium | Mid-level performers | Balanced learning |
| **advanced** | Advanced topics | Hard | High performers | Challenge & depth |

---

## 📊 Reward Function Design

**Rationale:**
- Immediate feedback (no waiting for future state)
- Direct alignment with student goal (improve score)
- Contextual to student's current performance level
- Symmetric: +1 for perfect improvement, -1 for perfect failure

**Formula:**
```
improvement = current_score - previous_score        # Range: [-100, +100]
reward = improvement / 100                         # Normalized: [-1, +1]
reward = clamp(reward, -1, +1)                     # Safety bounds
```

**Examples:**
- Student improved from 60 to 75: reward = +0.15
- Student maintained 70: reward = 0.0
- Student declined from 80 to 50: reward = -0.30

---

## 🔍 Structured Logging

All RL decisions logged with `[BANDIT]` prefix:

```
[BANDIT] Computing state: score=75.0, weak_count=2, state_id=medium-2
[BANDIT] Previous avg_score=70.0, prev_state_id=medium-1
[BANDIT] Selected action=mixed for state_id=medium-2
[BANDIT] ACTION=mixed → topics=['async_programming', 'error_handling'], difficulty=medium
[BANDIT] Course created: id=42, action=mixed, topics=[...], difficulty=medium
[BANDIT] Reward calculation: 75.0 - 70.0 = +5.0 → reward=0.050
[BANDIT] Q-value updated: state=medium-1, action=mixed, reward=0.050
[BANDIT] COMPLETE: state=medium-2, action=mixed, reward=0.050, course_id=42
```

---

## ✅ Key Guarantees

1. **Course Always Generated**: Primary attempt + fallback to revision action
2. **Action Determines Course**: Strict mapping prevents LLM from overriding
3. **Explicit LLM Instructions**: Topics and difficulty non-negotiable
4. **Simple Reward Signal**: Immediate feedback based on score improvement
5. **Bandit Learns**: Q-values accumulate to optimize action selection over time
6. **Response Consistency**: All required fields present in all scenarios

---

## 🚀 System Benefits

| Benefit | Impact |
|---------|--------|
| **Deterministic Mapping** | LLM can't override action → course strategy |
| **Always-On** | Fallback ensures course generation never fails |
| **Immediate Feedback** | Bandit learns from immediate score improvement |
| **Clear Causality** | State → Action → Reward → Learning |
| **Debugging** | Comprehensive logging shows every decision |

---

## 📝 Integration Notes

**Files Modified:**
1. `create_course_internal()` - Added topics + difficulty parameters
2. `/api/interview/evaluate` - Refactored RL section (10-step process)

**No Changes Needed:**
- Database schema (backward compatible)
- Bandit implementation (unchanged)
- API response format (extended, not broken)
- Error handling (enhanced, not removed)

---

## 🔗 Complete Flow Example

**Scenario:** Student completes 5-question interview

1. **Score Computation**: overall = 72, weak_topics = ["async", "concurrency"]
2. **State**: state_id = "medium-2" (from score 72, count 2)
3. **Bandit**: Q-values exist for all actions; selects "mixed" (highest Q)
4. **Mapping**: action="mixed" → topics=["async", "concurrency", "threading"], difficulty="medium"
5. **Course**: LLM generates course with 4-5 modules, all marked "medium" difficulty
6. **Reward**: previous_avg=65, current=72, improvement=7 → reward=0.07
7. **Update**: Q[medium-2]["mixed"] += 0.07 to running average
8. **Response**: 
   ```json
   {
     "new_course": {"course_id": 42, "title": "mid Developer - mixed Course"},
     "recommended_action": "mixed",
     "rl_metrics": {...reward details...}
   }
   ```

---

**Adaptive Loop Complete** ✅
Interview → Analysis → Bandit Decision → Course Generation → Reward Feedback → Learning

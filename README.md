# 🎧 OpenEnv: Helpdesk Triage & Resolution Environment

An OpenEnv-compliant simulation of a real-world e-commerce customer support workflow. 

## 🌍 Environment Description & Motivation

While many AI agent benchmarks focus on web-browsing or game-playing, real-world enterprise value lies in **workflow automation**. This environment simulates a multi-step Customer Support workflow. 

To succeed, an agent cannot simply be a reactive chatbot; it must act as a fully autonomous worker. It must:
1. **Read and comprehend** customer tickets.
2. **Utilize distinct tools** (Knowledge Base search, Database queries).
3. **Maintain stateful memory** of past actions (e.g., remembering a policy after searching for it).
4. **Perform basic mathematical reasoning** (calculating percentage-based refunds).
5. **Take irreversible actions** (issuing refunds, routing tickets, sending replies).

This serves as a rigorous, non-toy benchmark for an LLM's ability to combine tool-use, state tracking, and instruction following within a fixed step limit.

---

## 🎯 Task Descriptions & Difficulty

The environment features three graded tasks (scored `0.0` to `1.0`), escalating in complexity:

| Task Level | Objective | Expected Agent Behavior |
| :--- | :--- | :--- |
| **Easy** | Route a simple 'forgot password' ticket. | The agent must identify the intent and use the `route_ticket` action with the correct parameter (`"IT"`). Tests basic classification and single-tool use. |
| **Medium** | Answer a question about the return policy. | The agent must `search_kb` to retrieve the policy, store the answer in its context, and `draft_reply` accurately stating the 30-day policy. Tests memory and multi-step tool chaining. |
| **Hard** | Handle a damaged item complaint. | The agent must `check_order` to find the total order value ($1000), calculate a 20% partial refund, `issue_refund` for the exact calculated amount ($200), and `draft_reply` with an apology. Tests mathematical reasoning, DB lookups, and strict rule-following. |

---

## 🧩 Action & Observation Spaces

### Action Space
Agents must output strict JSON matching this Pydantic schema:

```json
{
  "action_type": "string",
  "parameters": {}
}
```
**Available `action_type` values:**
* `search_kb` (no params)
* `check_order` (no params)
* `issue_refund` (requires `{"amount": int}`)
* `draft_reply` (requires `{"text": string}`)
* `route_ticket` (requires `{"department": string}`)

### Observation Space
After every step, the agent receives an updated state:
```json
{
  "current_ticket": {
    "id": "T3", 
    "text": "My laptop arrived with a scratched screen...", 
    "customer_id": "C125", 
    "order_value": 1000
  },
  "action_result": "Order DB: Customer C125 spent $1000.",
  "system_messages": "Step 1/10"
}
```

---

## 🚀 Setup & Usage Instructions

### 1. Local Development (Python)
1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install groq python-dotenv
   ```
2. Create a `.env` file in the root directory and add your API key:
   ```text
   GROQ_API_KEY=gsk_your_key_here
   ```
3. Start the FastAPI server:
   ```bash
   uvicorn src.api:app --host 0.0.0.0 --port 8000
   ```

### 2. Docker Execution
To run the environment in an isolated container (mimicking the Hugging Face Space deployment):
```bash
docker build -t openenv-helpdesk .
docker run -p 8000:7860 -e GROQ_API_KEY="your_api_key_here" openenv-helpdesk
```

---

## 📊 Baseline Inference & Scores

This project includes a baseline evaluation script (`src/baseline.py`) powered by Groq's `llama-3.3-70b-versatile` model. It demonstrates that frontier open-weight models can successfully navigate the environment when provided with context history.

To run the baseline evaluation against the active server:
```bash
python src/baseline.py
```

### Official Baseline Scores:
```json
{
  "easy": 1.0,
  "medium": 1.0,
  "hard": 1.0
}
```
```
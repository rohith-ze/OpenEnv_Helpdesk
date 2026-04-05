import os
import requests
from groq import Groq
import json
from dotenv import load_dotenv

load_dotenv()
client = Groq()
API_URL = "http://localhost:8000"

def run_task(task_level):
    print(f"\n--- Starting Task: {task_level.upper()} ---")
    requests.post(f"{API_URL}/reset?task_level={task_level}")
    
    tasks_info = requests.get(f"{API_URL}/tasks").json()
    task_description = tasks_info["tasks"][task_level]["description"]
    print(f"Objective: {task_description}")
    
    # Get the initial state
    initial_state = requests.get(f"{API_URL}/state").json()
    
    # Set up the SYSTEM instructions once
    # Set up the SYSTEM instructions once
    messages = [
        {
            "role": "system", 
            "content": f"""You are an AI customer support agent. 
Your Objective: {task_description}

Initial State: {json.dumps(initial_state)}

CRITICAL RULES:
1. You must solve the task step-by-step.
2. DO NOT take unprompted actions. If the objective says to route a ticket, ONLY route the ticket. Do not draft polite replies unless explicitly asked.
3. Output ONLY a raw JSON action object. Do not include markdown formatting, backticks, or conversational text.
Format: {{"action_type": "...", "parameters": {{...}}}}

Available actions: 
- search_kb (no parameters needed)
- check_order (no parameters needed)
- issue_refund (requires integer "amount" parameter)
- draft_reply (requires string "text" parameter)
- route_ticket (requires string "department" parameter)
"""
        }
    ]
    
    for step in range(10):
        try:
            # We pass the ENTIRE message history to the model now
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.1,
                max_completion_tokens=1024,
                top_p=1,
                stream=False,
                stop=None
            )
            
            response_text = completion.choices[0].message.content.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith("```"):
                response_text = response_text[3:-3].strip()
                
            action_dict = json.loads(response_text)
            print(f"Step {step + 1} Action: {action_dict['action_type']} | Params: {action_dict.get('parameters', {})}")
            
            # Add the agent's action to the history
            messages.append({"role": "assistant", "content": json.dumps(action_dict)})
            
            # Execute the step
            step_res = requests.post(f"{API_URL}/step", json=action_dict).json()
            obs_result = step_res['observation']['action_result']
            print(f"Result: {obs_result}")
            
            # Add the environment's result to the history so the agent knows what happened!
            messages.append({"role": "user", "content": f"Observation: {obs_result}"})
            
            if step_res["reward"]["is_done"]:
                final_score = step_res["reward"]["total_score"]
                print(f"Task Completed! Final Score: {final_score}")
                return final_score
                
        except json.JSONDecodeError:
            print(f"Error: Agent output invalid JSON -> {response_text}")
            return 0.0
        except Exception as e:
            print(f"Error during execution: {e}")
            return 0.0
            
    print("Agent ran out of steps!")
    return 0.0 

if __name__ == "__main__":
    print("Running Groq Baseline Agent...")
    scores = {
        "easy": run_task("easy"),
        "medium": run_task("medium"),
        "hard": run_task("hard")
    }
    print("\n====================")
    print("FINAL BASELINE SCORES:")
    print(json.dumps(scores, indent=2))
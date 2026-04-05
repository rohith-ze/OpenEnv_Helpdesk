import copy
from .models import Action, Observation, Reward

TASKS = {
    "easy": {
        "id": "easy_1",
        "description": "Route a simple 'forgot password' ticket to the IT department.",
        "expected_difficulty": "easy",
        "ticket": {"id": "T1", "text": "I can't log into my account.", "customer_id": "C123"}
    },
    "medium": {
        "id": "medium_1",
        "description": "Search the Knowledge Base for the return policy and draft a reply to the customer.",
        "expected_difficulty": "medium",
        "ticket": {"id": "T2", "text": "How many days do I have to return my shoes?", "customer_id": "C124"}
    },
    "hard": {
        "id": "hard_1",
        "description": "Verify customer order history. They received a damaged item. Issue a 20% partial refund and draft an apology.",
        "expected_difficulty": "hard",
        "ticket": {"id": "T3", "text": "My laptop arrived with a scratched screen. I want compensation.", "customer_id": "C125", "order_value": 1000}
    }
}

class HelpdeskEnv:
    def __init__(self, task_level="medium"):
        self.task_level = task_level
        self.reset()

    def reset(self) -> Observation:
        self.current_task = copy.deepcopy(TASKS[self.task_level])
        self.state_data = {
            "ticket": self.current_task["ticket"],
            "kb_searched": False,
            "order_checked": False,
            "refund_issued": 0,
            "reply_drafted": None,
            "routed_to": None,
            "step_count": 0
        }
        return self.state()

    def state(self) -> Observation:
        return Observation(
            current_ticket=self.state_data["ticket"],
            action_result="Awaiting action.",
            system_messages="System normal."
        )

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        self.state_data["step_count"] += 1
        reward_val = 0.0
        done = False
        action_result = ""

        # Action Execution & Reward Shaping (Partial Progress)
        if action.action_type == "search_kb":
            self.state_data["kb_searched"] = True
            action_result = "KB Result: Return policy is 30 days."
            reward_val = 0.2  # Positive reinforcement for gathering info

        elif action.action_type == "check_order":
            self.state_data["order_checked"] = True
            action_result = f"Order DB: Customer {self.state_data['ticket']['customer_id']} spent $1000."
            reward_val = 0.2

        elif action.action_type == "route_ticket":
            self.state_data["routed_to"] = action.parameters.get("department", "unknown")
            action_result = f"Ticket routed to {self.state_data['routed_to']}."
            done = True

        elif action.action_type == "issue_refund":
            if not self.state_data["order_checked"]:
                reward_val = -0.5 # Penalty for blind refunds
                action_result = "ERROR: Must check order history before refunding."
            else:
                amount = action.parameters.get("amount", 0)
                self.state_data["refund_issued"] = amount
                action_result = f"Refunded ${amount}."

        elif action.action_type == "draft_reply":
            self.state_data["reply_drafted"] = action.parameters.get("text", "")
            action_result = "Reply drafted."
            done = True

        else:
            reward_val = -0.1 # Penalty for hallucinated actions
            action_result = "Invalid action."

        # Cap steps to prevent infinite loops
        if self.state_data["step_count"] >= 10:
            done = True

        obs = Observation(
            current_ticket=self.state_data["ticket"],
            action_result=action_result,
            system_messages=f"Step {self.state_data['step_count']}/10"
        )
        
        # Calculate final grader score if done
        final_score = self._run_grader() if done else 0.0
        if done: reward_val += final_score

        reward = Reward(step_reward=reward_val, total_score=final_score, is_done=done, info={})
        return obs, reward, done, self.state_data

    def _run_grader(self) -> float:
        """Deterministic grading logic (0.0 to 1.0)"""
        score = 0.0
        if self.task_level == "easy":
            if self.state_data["routed_to"] == "IT": score = 1.0
            
        elif self.task_level == "medium":
            if self.state_data["kb_searched"] and self.state_data["reply_drafted"]:
                if "30 days" in self.state_data["reply_drafted"].lower():
                    score = 1.0
                else:
                    score = 0.5 # Partial credit for replying without specific policy
                    
        elif self.task_level == "hard":
            if self.state_data["order_checked"] and self.state_data["reply_drafted"]:
                if self.state_data["refund_issued"] == 200: # 20% of 1000
                    score = 1.0
                elif self.state_data["refund_issued"] > 0:
                    score = 0.5 # Wrong amount
        return score
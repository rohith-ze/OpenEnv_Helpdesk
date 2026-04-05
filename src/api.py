from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.env import HelpdeskEnv, TASKS
from src.models import Action
import subprocess

app = FastAPI(title="OpenEnv Helpdesk", version="1.0.0")
current_env = HelpdeskEnv()

@app.get("/state")
def get_state():
    return current_env.state()

@app.post("/reset")
def reset_env(task_level: str = "medium"):
    if task_level not in TASKS:
        raise HTTPException(status_code=400, detail="Invalid task level")
    current_env.__init__(task_level)
    return current_env.state()

@app.post("/step")
def step_env(action: Action):
    obs, reward, done, info = current_env.step(action)
    return {"observation": obs.model_dump(), "reward": reward.model_dump()}

@app.get("/tasks")
def get_tasks():
    return {
        "tasks": TASKS,
        "action_schema": Action.model_json_schema()
    }

@app.get("/grader")
def get_grader_score():
    return {"score": current_env._run_grader()}

@app.post("/baseline")
def run_baseline():
    """Triggers the inference script to generate reproducible scores."""
    try:
        result = subprocess.run(["python", "src/baseline.py"], capture_output=True, text=True)
        return {"output": result.stdout, "errors": result.stderr}
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))
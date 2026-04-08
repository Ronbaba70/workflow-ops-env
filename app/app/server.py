from fastapi import FastAPI
from app.env import WorkFlowOpsEnv
from app.models import Action

app = FastAPI()
env = WorkFlowOpsEnv()

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/reset")
def reset():
    return env.reset()

@app.post("/step")
def step(action: Action):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
def state():
    return env.state()

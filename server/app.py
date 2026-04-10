from fastapi import FastAPI
from app.env import WorkFlowOpsEnv
from app.models import Action
import uvicorn

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/reset")
def reset():
    env = WorkFlowOpsEnv()
    return env.reset()

@app.post("/step")
def step(action: Action):
    env = WorkFlowOpsEnv()
    obs, reward, done, info = env.step(action)
    return {
        "obs": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

# ✅ THIS IS REQUIRED
def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

# ✅ ALSO REQUIRED
if __name__ == "__main__":
    main()

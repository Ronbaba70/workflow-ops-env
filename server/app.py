from fastapi import FastAPI
from app.env import WorkFlowOpsEnv
from app.models import Action

app = FastAPI()
env = WorkFlowOpsEnv()


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/reset")
def reset():
    obs = env.reset()

    return {
        "task_id": obs.task_id,
        "step_count": obs.step_count,
        "visible_emails": obs.visible_emails,
        "code_snippet": obs.code_snippet,
        "data_sample": obs.data_sample,
        "instruction": obs.instruction
    }


@app.post("/step")
def step(action: dict):
    action_obj = Action(**action)

    obs, reward, done, info = env.step(action_obj)

    return {
        "obs": {
            "task_id": obs.task_id,
            "step_count": obs.step_count,
            "visible_emails": obs.visible_emails,
            "code_snippet": obs.code_snippet,
            "data_sample": obs.data_sample,
            "instruction": obs.instruction
        },
        "reward": reward.score,
        "done": done,
        "info": info
    }


# 🔥 ADD THIS (CRITICAL)
def main():
    return app


# 🔥 ALSO ADD THIS (CRITICAL)
if __name__ == "__main__":
    main()

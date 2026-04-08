import os, json
from openai import OpenAI
from app.env import WorkFlowOpsEnv
from app.models import Action

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("API_BASE_URL")
)

MODEL = os.getenv("MODEL_NAME")

env = WorkFlowOpsEnv()

def safe_parse(txt):
    try:
        return json.loads(txt)
    except:
        return {"action_type": "submit", "payload": {}}

def run():
    print("[START]")
    total = 0

    for _ in range(3):
        obs = env.reset()
        done = False
        memory = []

        while not done:
            prompt = f"Task: {obs.instruction}\nObs: {obs.json()}\nHistory: {memory}\nReturn JSON."

            res = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}]
            )

            action_json = safe_parse(res.choices[0].message.content)
            action = Action(**action_json)

            obs, reward, done, info = env.step(action)
            memory.append(action_json)

            print(f"[STEP] reward={reward.score} task={info['task']}")

        total += reward.score

    print(f"[END] final_score={total/3}")

if __name__ == "__main__":
    run()

import random
from app.models import Observation, Action, Reward
from app.tasks import load_task, sample_task
from app.graders import grade_task

class WorkFlowOpsEnv:

    def __init__(self):
        self.max_steps = 12
        self.reset()

    def reset(self):
        self.task = sample_task()
        self.state_data = load_task(self.task)
        self.step_count = 0
        return self._obs()

    def state(self):
        return self.state_data

    def step(self, action: Action):
        self.step_count += 1

        reward = 0.0
        reason = ""

        if self.task == "email":
            if action.action_type == "classify":
                i = action.payload["id"]
                label = action.payload["label"]
                self.state_data["labels"][i] = label
                reward += 0.2

        elif self.task == "code":
            if action.action_type == "edit":
                self.state_data["code"] = action.payload.get("patch", "")
                reward += 0.2
            if action.action_type == "run":
                if "arr[i]" in self.state_data["code"]:
                    reward += 0.3

        elif self.task == "data":
            if action.action_type == "clean":
                self.state_data["data"] = action.payload.get("data", [])
                reward += 0.3

        if action.action_type == "submit":
            final = grade_task(self.task, self.state_data)
            reward += final
            reason = "final"
            done = True
        else:
            done = False

        if self.step_count >= self.max_steps:
            done = True

        reward = max(0.0, min(1.0, reward))

        return self._obs(), Reward(score=reward, reason=reason), done, {"task": self.task}

    def _obs(self):
        return Observation(
            task_id=self.task,
            step_count=self.step_count,
            visible_emails=list(self.state_data.get("emails", {}).keys())[:2],
            code_snippet=self.state_data.get("code"),
            data_sample=self.state_data.get("data", [])[:2],
            instruction=self.state_data["instruction"]
        )

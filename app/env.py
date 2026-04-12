import random
from app.models import Observation, Action, Reward
from app.tasks import load_task, sample_task
from app.graders import grade_task


class WorkFlowOpsEnv:

    def __init__(self):
        self.max_steps = 12
        self.reset()

    # ---------------- RESET ----------------
    def reset(self):
        self.task = sample_task()
        self.state_data = load_task(self.task)
        self.step_count = 0

        self.progress = {
            "email_done": set(),
            "data_cleaned": False,
            "code_fixed": False
        }

        return self._obs()

    # ---------------- STATE ----------------
    def state(self):
        return self.state_data

    # ---------------- STEP ----------------
    def step(self, action: Action):
        self.step_count += 1

        reward = 0.0
        reason = ""
        done = False

        # ---------------- EMAIL ----------------
        if self.task == "email":

            if action.action_type == "classify":
                i = action.payload.get("id")
                label = action.payload.get("label")

                if i in self.state_data["emails"]:
                    true_label = self.state_data["emails"][i].get("label", "spam")

                    if i not in self.progress["email_done"]:
                        if label == true_label:
                            reward += 0.3
                        else:
                            reward -= 0.1

                        self.progress["email_done"].add(i)

                    self.state_data["labels"][i] = label

        # ---------------- DATA ----------------
        elif self.task == "data":

            if action.action_type == "clean":
                new_data = action.payload.get("data", [])
                old_data = self.state_data["data"]

                if len(new_data) < len(old_data):
                    reward += 0.2

                if all(x["value"] is not None for x in new_data):
                    reward += 0.3

                if not self.progress["data_cleaned"]:
                    self.progress["data_cleaned"] = True
                else:
                    reward *= 0.5

                self.state_data["data"] = new_data

        # ---------------- CODE ----------------
        elif self.task == "code":

            if action.action_type == "edit":
                new_code = action.payload.get("patch", "")
                self.state_data["code"] = new_code

                # partial fix
                if "range(len(arr)-1)" in new_code:
                    reward += 0.1

                # correct fix
                if "arr[i]" in new_code and "i+1" not in new_code:
                    reward += 0.4

                if not self.progress["code_fixed"]:
                    self.progress["code_fixed"] = True
                else:
                    reward *= 0.5

            if action.action_type == "run":
                if "arr[i]" in self.state_data["code"]:
                    reward += 0.3

        # ---------------- SUBMIT (GLOBAL) ----------------
        if action.action_type == "submit":
            final = grade_task(self.task, self.state_data)
            reward += final
            reason = "final"
            done = True

        # ---------------- MAX STEP ----------------
        if self.step_count >= self.max_steps:
            done = True

        reward = max(0.0, min(1.0, reward))

        return (
            self._obs(),
            Reward(score=reward, reason=reason),
            done,
            {"task": self.task}
        )

    # ---------------- OBS ----------------
    def _obs(self):
        return Observation(
            task_id=self.task,
            step_count=self.step_count,
            visible_emails=list(self.state_data.get("emails", {}).keys()),
            code_snippet=self.state_data.get("code"),
            data_sample=self.state_data.get("data", []),
            instruction=self.state_data["instruction"]
        )

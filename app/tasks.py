import random

def load_task(task_type):

    if task_type == "email":
        return {
            "instruction": "Classify emails using tools",
            "emails": {
                1: {"text": "URGENT: reset password", "sender": "it@company.com"},
                2: {"text": "Win a free iPhone!!!", "sender": "spam@fake.com"},
                3: {"text": "Meeting at 5pm", "sender": "boss@company.com"}
            },
            "labels": {}
        }

    if task_type == "code":
        return {
            "instruction": "Fix all bugs until tests pass",
            "code": "for i in range(len(arr)): total += arr[i+1]",
            "tests_passed": False
        }

    if task_type == "data":
        return {
            "instruction": "Clean dataset using tools",
            "data": [
                {"id": 1, "value": 10},
                {"id": 1, "value": 10},
                {"id": 2, "value": None}
            ]
        }

def sample_task():
    return random.choice(["email", "code", "data"])

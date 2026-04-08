def grade_email(state):
    correct = {1: "urgent", 2: "spam", 3: "normal"}
    score = sum(state["labels"].get(k) == v for k, v in correct.items())
    return max(0.0, score / len(correct))

def grade_code(state):
    if "arr[i]" in state["code"] and "i+1" not in state["code"]:
        return 1.0
    elif "arr[i]" in state["code"]:
        return 0.5
    return 0.0

def grade_data(state):
    data = state["data"]
    if len(data) == 1 and data[0]["id"] == 1:
        return 1.0
    elif len(data) == 2:
        return 0.5
    return 0.0

def grade_task(task, state):
    if task == "email":
        return grade_email(state)
    if task == "code":
        return grade_code(state)
    if task == "data":
        return grade_data(state)

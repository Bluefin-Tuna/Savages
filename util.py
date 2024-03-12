from collections import defaultdict

PROMPT_PATH = "./data"
agents = []

def group(part, k):
    groups = []
    for g in part:
        for i in range(len(part[g]) - k):
            groups.append(part[g][i:i+k])
    return groups

def partition(examples, key):
    part = defaultdict(list)
    for example in examples:
        if key not in example:
            continue
        part[example[key]].append(example)
    return part

def load_prompt(prompt, _path = PROMPT_PATH):
    file = open(f"{_path}/{prompt}.prompt", "r")
    rprompt = file.read()
    file.close()
    return rprompt

def contaminate():

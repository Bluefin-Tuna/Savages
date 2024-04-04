from collections import defaultdict
from prompt import Prompt
from langchain.chat_models import ChatOpenAI, ChatAnthropic
import jsonlines

DATA_PATH = "./data"

agents = []

def group(data, k):
    groups = []
    for g in data:
        for i in range(len(data[g]) - k):
            groups.append(data[g][i:i+k])
    return groups

def contaminate(example):
    pass

class Dataset:

    def __init__(self, data="mixed.jsonl"):
        self.data = []
        with jsonlines.open(f"{DATA_PATH}/{data}", mode="r") as reader:
            self.data = [li for li in reader][0]
        self.authors = set()
        self.schools = set()
        for ex in self.data:
            self.authors.add(ex["author"])
            self.schools.add(ex["school"])

    def partition(self, key):
        part = defaultdict(list)
        for example in self.data:
            if key not in example:
                continue
            part[example[key]].append(example)
        return part

agents = [
    ChatOpenAI(openai_api_key="", model="gpt-3.5-turbo-16k", temperature=0.5)
]

dataset = Dataset()
for ex in dataset.data[0:1]:
    print(ex)
    kk = Prompt().contaminate(ex, agent=agents[0])
    print(kk)
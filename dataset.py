from collections import defaultdict
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


def contaminate():
    
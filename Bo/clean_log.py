import os
import re
import json
from collections import defaultdict

def process_log(fp):
    num_options = int(re.split("[_\.]", fp)[1])
    log = open(fp, "r").read().strip()
    pattern = "file: (.*?), Author: (.*?), Title: (.*?), Accuracy: ([0-9.]+)"
    results = list(re.findall(pattern, log))
    return num_options, results

def main(fps):
    log_data = defaultdict(list)
    for fp in fps:
        num_options, results = process_log(fp)
        for result in results:
            passage_type = re.split("[_\.]", result[0])[-2].replace(".jsonl", "")
            log_data[result[2]].append({
                "num_options": num_options,
                "passage_type": passage_type,
                "author": result[1],
                "accuracy": float(result[3])
            })
    return dict(log_data)

if __name__ == "__main__":
    fps = os.listdir(".")
    logs = [fp for fp in fps if fp.endswith(".exp.log")]
    log_data = main(logs)
    print(log_data)
    json.dump(log_data, open("log.data", "w+"), indent=4)    
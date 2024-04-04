import os
import random
import time

random.seed(0)

PROMPT_PATH = "./prompt"


def load_file(fp):
    f = open(fp, "r")
    text = f.read()
    f.close()
    return text

def load_prompt(prompt, _path = PROMPT_PATH):
    fp = f"{PROMPT_PATH}/{prompt}.prompt"
    text = load_file(fp)
    return text

def load_probe_prompt(prompt):
    return load_prompt(f"probe.{prompt}")

def load_judge_prompt(prompt):
    return load_prompt(f"judge.{prompt}")

class Prompt:
    def __init__(self):
        # Goal
        self.goal = load_prompt("goal")

        # Data Contamination
        self.choice = [load_probe_prompt("choice"), load_judge_prompt("choice")]
        self.context = [load_probe_prompt("context"), load_judge_prompt("context")]
        self.paraphraseQuestions = [load_probe_prompt("paraphraseQ"), load_judge_prompt("paraphraseQ")]

        # Inference
        self.task = load_prompt("task")
    
    def create_inference_prompt(self, ex, with_permutation):
        if with_permutation:
            random.shuffle(ex["choices"])
        str_choices = "\n".join([f"{k}. {v}" for (k, v) in ex["choices"]])
        return (
            f"{self.task}\n\n"
            "```passage\n"
            f'{ex["sentence"]}\n'
            "```\n\n"
            f"{str_choices}")

    def _probe_context(self, ex):
        return (
            f"{self.context[0]}\n\n"
            "```context\n"
            f'{ex["sentence"]}\n'
            "```")

    def _probe_choice(self, ex):
        str_choices = "\n".join([f"{k}. {v}" for (k, v) in ex["choices"]])
        return (
            f"{self.choice[0]}\n\n"
            "```passage\n"
            f'{ex["sentence"]}\n'
            "```\n\n"
            "```choice\n"
            f"{str_choices}\n"
            "```")

    def _judge_context(self, ex, cex):
        return (
            f"{self.context[1]}\n\n"
            "```original\n"
            f'{ex["sentence"]}\n'
            "```\n\n"
            "```modified\n"
            f"{cex}\n"
            "```")

    def _judge_choice(self, ex, modifiedChoices):
        return (
            f"{self.choice[1]}\n\n"
            "```original\n"
            f"{ex['choices']}\n"
            "```\n\n"
            "```modified\n"
            f"{modifiedChoices}\n"
            "```")

    def contaminate_choice(self, ex, agent):
        i = 0
        while i < 3:
            _probe_pt = self._probe_choice(ex)
            modifiedChoices = agent.predict(_probe_pt)
            _judge_pt = self._judge_choice(ex, modifiedChoices)
            if "Yes" in _judge_pt.lower():
                modifiedExample = ex.copy()
                modifiedExample['choices'] = modifiedChoices  # Update the choices
                return modifiedExample
            i += 1
        return None


    def contaminate_context(self, ex, agent):
        i = 0
        while i < 3:
            _probe_pt = self._probe_context(ex)
            modifiedContext = agent.predict(_probe_pt)
            _judge_pt = self._judge_context(ex, modifiedContext.content)
            if "Yes" in _judge_pt.content.lower():
                modifiedExample = ex.copy()
                modifiedExample["sentence"] = modifiedContext.content
                return modifiedExample
            i += 1
        return None

    def contaminate(self, ex, agent):
        ex = self.contaminate_choice(ex, agent)
        ex = self.contaminate_context(ex, agent)
        return self.create_inference_prompt(ex, with_permutation=True)


import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import itertools
import random
import os
from torch.nn.functional import softmax


def main(config_path):

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Only one model is used in this file
    model_name = "meta-llama/Llama-2-7b-hf"
    device_id = 0
    # If device_id is not provided, set the device to 'auto'
    device = f'cuda:{device_id}' if device_id >= 0  else 'auto'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
    model.eval()
    first_device = next(model.parameters()).device if not device_id else device_id

    for method, config_options in config.items():
        keys, values = zip(*config_options.items())
        experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

        for experiment in experiments:
            print(f"Running {method} with configuration: {experiment}")
            run_experiment(tokenizer, model, first_device, **experiment)

            
def run_experiment(tokenizer, model, first_param_device, **experiment):
    
    seed = experiment['seed']
    # fix the random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    num_options = experiment['num_options']
    # Load authors from a JSON file
    with open('authors.json', 'r') as f:
        all_authors = json.load(f)["philosophers"]
    if num_options > len(all_authors):
        raise ValueError("Number of options cannot be greater than the number of authors.")
    if num_options < 2:
        raise ValueError("Number of options must be at least 2.")

    system_prompt = """ You will be presented with a multiple-choice question that quotes a passage from a book by a philosopher. 
    The possible answer choices will be the names of different philosophers who could potentially be the author of the quoted passage. 
    Your task is to carefully read the quoted passage and select the philosopher from the given choices who is the actual author of that passage from their book. 
    Do not provide any additional analysis or explanation - simply indicate which philosopher is the correct answer by choosing the corresponding option.
    Passage:
    """
    
    for filename in os.listdir('./data'):
        if filename.endswith(".jsonl"):
            data_file = os.path.join('./data', filename)
            correct_author = filename.split('_')[0]
            sentence_or_paragraph = filename.split('_')[-1].split('.')[0]
            book = filename.split('_')[1]
            prediction = []
            with open(data_file, 'r') as f:
                # Skip the first two lines, which are metadata
                next(f)
                next(f)
                for line in f:
                    data = json.loads(line)
                    text_content = data.get(sentence_or_paragraph)

                    # Create dynamic options list with the correct author included
                    options = random.sample([a for a in all_authors if a != correct_author], num_options - 1)
                    options.append(correct_author)
                    random.shuffle(options)
                    options_dict = {chr(65 + i): options[i] for i in range(len(options))}

                    option_prompt ="\n" + "Options: \n" + '\n'.join([f"{k}. {v}" for k, v in options_dict.items()]) + "\nAnswer:"

                    input_text = system_prompt + text_content + option_prompt
                    # LLama2 tokenizer will add batch_size dimension by default
                    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(first_param_device)

                    # Calculate the probabilities for each option
                    with torch.no_grad():
                        output_logits = model(input_ids)[0][:, -1, :]
                        output_probs = softmax(output_logits, dim=-1)[0]
                        option_tokens = [tokenizer.encode(option)[1] for option in options_dict.keys()]
                        option_probs = [output_probs[token_id] for token_id in option_tokens]
                        predicted_author = options_dict[max(options_dict, key=lambda k: output_probs[tokenizer.encode(k)[1]].item())]

                    prediction.append(predicted_author==correct_author)
                    if len(prediction) % 10 == 0:
                        print(f"Processed passage: {len(prediction)}")

            accuracy = prediction.count(True) / len(prediction)
            print(f"Author: {correct_author}, Title: {book}, Accuracy: {accuracy:.2f}")
    

# def run_experiment(tokenizer, model, first_param_device, **experiment):

#     prediction  = []
#     lookup_table = {
#     'Aristotle' : 'A',
#     'Arthur Schopenhauer' : 'B',
#     'Friedrich Wilhelm Nietzsche' : 'C',
#     'Georg Wilhelm Friedrich Hegel' : 'D',
#     'Immanuel Kant' : 'E',
#     'Ludwig Feuerbach' : 'F',
#     'Plato' : 'G'
# }
#     options = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
#     right_answer = lookup_table[ data_file.split('_')[0]]
#     sentence_or_paragraph = data_file.split('_')[-1].split('.')[0]
#     author = data_file.split('_')[0]
#     book = data_file.split('_')[1]
#     option_tokens = [tokenizer.encode(option)[1] for option in options]
#     # Prepare the input text
#     system_prompt = """ You will be presented with a multiple-choice question that quotes a passage from a book by a philosopher. 
#     The possible answer choices will be the names of different philosophers who could potentially be the author of the quoted passage. 
#     Your task is to carefully read the quoted passage and select the philosopher from the given choices who is the actual author of that passage from their book. 
#     Do not provide any additional analysis or explanation - simply indicate which philosopher is the correct answer by choosing the corresponding option.
#     Passage: """
#     option_prompt = """
#     Options: 
#     A. Aristotle
#     B. Arthur Schopenhauer
#     C. Friedrich Wilhelm Nietzsche
#     D. Georg Wilhelm Friedrich Hegel
#     E. Immanuel Kant
#     F. Ludwig Feuerbach
#     G. Plato
#     Answer:"""
#     # data_file = "/home/byuan48/models/CS8803/Immanuel Kant_Kant's Critique of Judgement_sentence.jsonl"
#     # Open the jsonl file
#     with open(data_file, 'r') as f:
#         # Skip the first two lines
#         next(f)
#         next(f)
#         for line in f:
#             data = json.loads(line)
#             text_content = data.get(sentence_or_paragraph)
#             input_text = system_prompt + text_content + option_prompt
#             # print(f"length:{len(text_content)}")
#             # Tokenize the input
#             input_ids = tokenizer.encode(input_text, return_tensors="pt").to(first_param_device)
#             # Find the token IDs for the options
#             # Calculate the probabilities for each option
#             with torch.no_grad():
#                 output_logits = model(input_ids)[0][:, -1, :]
#                 output_probs = torch.softmax(output_logits, dim=-1)[0]
#                 max_prob_idx = output_probs.argmax().item()
#                 max_prob_word = tokenizer.decode([max_prob_idx])
#                 option_probs = [output_probs[token_id].item() for token_id in option_tokens]
#             prediction.append( options[np.argmax(option_probs)])
#             # print(f"The token with the highest probability is: {max_prob_word} (index: {max_prob_idx})")
#             # print(f"The current prediction is : {prediction[-1]}")
#             if len(prediction) % 10 == 0:
#                 print(f"Precessed passage: {len(prediction)}\n")
#             # for option, prob in zip(options, option_probs):
#             #     print(f"Probability of predicting {option}: {prob:.4f}")
#     print(f"author:{author} title:{book} model:{model_name}")
#     print(f"accu:{prediction.count(right_answer) / len(prediction)}")


# Main processing loop
if __name__ == "__main__":
    
    main('config.json')
    
    
    







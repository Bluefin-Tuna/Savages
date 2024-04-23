import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import matplotlib.pyplot as plt
import json

# Setup command-line argument parsing
parser = argparse.ArgumentParser(description='Process input parameters for the text perturbation and uncertainty measurement script.')
# available model names: "huggyllama/llama-7b"
parser.add_argument('--model', type=str, default="meta-llama/Llama-2-13b-chat-hf", help='HuggingFace Model used for inference')
parser.add_argument('--input_file_name', type=str, default="Immanuel Kant_Kant's Critique of Judgement_sentence.jsonl", help='data file')
args = parser.parse_args()
model_name = args.model
data_file = args.input_file_name

# Main processing loop
if __name__ == "__main__":

    # Set the device to GPU if available
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
    model.eval()
    first_param_device = next(model.parameters()).device
    prediction  = []
    lookup_table = {
    'Aristotle' : 'A',
    'Arthur Schopenhauer' : 'B',
    'Friedrich Wilhelm Nietzsche' : 'C',
    'Georg Wilhelm Friedrich Hegel' : 'D',
    'Immanuel Kant' : 'E',
    'Ludwig Feuerbach' : 'F',
    'Plato' : 'G',
    'Jennifer Saul' : 'H',
        
}
    options = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    right_answer = lookup_table[ data_file.split('_')[0]]
    sentence_or_paragraph = data_file.split('_')[-1].split('.')[0]
    author = data_file.split('_')[0]
    book = data_file.split('_')[1]
    option_tokens = [tokenizer.encode(option)[1] for option in options]
    # Prepare the input text
    system_prompt = """ You will be presented with a multiple-choice question that quotes a passage from a book by a philosopher. 
    The possible answer choices will be the names of different philosophers who could potentially be the author of the quoted passage. 
    Your task is to carefully read the quoted passage and select the philosopher from the given choices who is the actual author of that passage from their book. 
    Do not provide any additional analysis or explanation - simply indicate which philosopher is the correct answer by choosing the corresponding option.
    Passage: """
    option_prompt = """
    Options: 
    A. Aristotle
    B. Arthur Schopenhauer
    C. Friedrich Wilhelm Nietzsche
    D. Georg Wilhelm Friedrich Hegel
    E. Immanuel Kant
    F. Ludwig Feuerbach
    G. Plato
    Answer:"""
    # data_file = "/home/byuan48/models/CS8803/Immanuel Kant_Kant's Critique of Judgement_sentence.jsonl"
    # Open the jsonl file
    with open(data_file, 'r') as f:
        # Skip the first two lines
        next(f)
        next(f)
        for line in f:
            data = json.loads(line)
            text_content = data.get(sentence_or_paragraph)
            input_text = system_prompt + text_content + option_prompt
            # print(f"length:{len(text_content)}")
            # Tokenize the input
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(first_param_device)
            # Find the token IDs for the options
            # Calculate the probabilities for each option
            with torch.no_grad():
                output_logits = model(input_ids)[0][:, -1, :]
                output_probs = torch.softmax(output_logits, dim=-1)[0]
                max_prob_idx = output_probs.argmax().item()
                max_prob_word = tokenizer.decode([max_prob_idx])
                option_probs = [output_probs[token_id].item() for token_id in option_tokens]
            prediction.append( options[np.argmax(option_probs)])
            # print(f"The token with the highest probability is: {max_prob_word} (index: {max_prob_idx})")
            # print(f"The current prediction is : {prediction[-1]}")
            if len(prediction) % 10 == 0:
                print(f"Precessed passage: {len(prediction)}\n")
            # for option, prob in zip(options, option_probs):
            #     print(f"Probability of predicting {option}: {prob:.4f}")
    print(f"author:{author} title:{book} model:{model_name}")
    print(f"accu:{prediction.count(right_answer) / len(prediction)}")
    
   





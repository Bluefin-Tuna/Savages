import json
import random
import nltk
from nltk.tokenize import sent_tokenize
import re
import os

nltk.download('punkt')  # Ensure the 'punkt' tokenizer is downloaded



# Get the author and title 
def extract_author_title(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Pattern to match the title line
    title_pattern = r'Title:\s*(.+)'
    # Pattern to match the author line
    author_pattern = r'Author:\s*(.+)'

    title = None
    author = None

    for line in lines:
        # Check for the title line
        title_match = re.search(title_pattern, line)
        if title_match:
            title = title_match.group(1).strip()

        # Check for the author line
        author_match = re.search(author_pattern, line)
        if author_match:
            author = author_match.group(1).strip()

        # If both title and author are found, exit the loop
        if title and author:
            break

    return title, author


# Function to split text into sentences and calculate their locations
def split_into_sentences_with_location(text):
    # Use NLTK to split the text into sentences
    raw_sentences = sent_tokenize(text)
    sentences = []
    locations = []
    location = 0

    for raw_sentence in raw_sentences:
        start_location = text.find(raw_sentence, location)  # Find the start location of the sentence
        sentence = raw_sentence.replace("\n", " ").strip()  # Normalize spaces and newlines
        if len(sentence) > 20:  # Only include sentences longer than 20 characters: make the sentences meaningful and remove corporas like "Section V."
            sentences.append(sentence)
            locations.append(start_location)
        location = start_location + len(sentence)  # Update location to search for the next sentence

    # Combine sentences with their locations
    sentences_with_locations = [{"sentence": sen, "location": loc} for sen, loc in zip(sentences, locations)]
    return sentences_with_locations
# Function to split text into paragraphs and calculate their locations
def split_into_paragraphs_with_location(text):
    # Split the text into paragraphs based on two consecutive newlines
    raw_paragraphs = re.split(r'\n\n', text)

    paragraphs = []
    locations = []
    location = 0

    for raw_paragraph in raw_paragraphs:
        start_location = text.find(raw_paragraph, location)
        paragraph = raw_paragraph.strip()  # Remove leading/trailing whitespace

        if len(paragraph) > 50:  # Include non-empty paragraphs
            paragraphs.append(paragraph)
            locations.append(start_location)

        location = start_location + len(raw_paragraph) + 2  # Update location, accounting for \n\n

    # Combine paragraphs with their locations
    paragraphs_with_locations = [{"paragraph": para, "location": loc} for para, loc in zip(paragraphs, locations)]

    return paragraphs_with_locations

def run(txt_file_path):

    title, author = extract_author_title(txt_file_path)
    print(f"Title: {title}")
    print(f"Author: {author}")
    
    # Load and parse the TXT content

    with open(txt_file_path, 'r', encoding='utf-8') as file:
        txt_content = file.read()
    # Split the extracted text into sentences(pargraphs) with locations
    all_sentences_with_location = split_into_sentences_with_location(txt_content)
    all_paragraphs_with_location = split_into_paragraphs_with_location(txt_content)


    # Randomly select 1000 sentences(100 paragraphs) from the list, ensuring the list is large enough
    randomly_selected_sentences_with_location = random.sample(all_sentences_with_location, min(1000, len(all_sentences_with_location)))
    randomly_selected_paragraphs_with_location = random.sample(all_paragraphs_with_location, min(100, len(all_paragraphs_with_location)))

    author_and_title = f"{author}_" + f"{title}_"

    txt_output_file_path = author_and_title + 'sentence.jsonl'
    with open(txt_output_file_path, 'w', encoding='utf-8') as outfile:
        json.dump(f"Title: {title}", outfile)
        outfile.write('\n')
        json.dump(f"Author: {author}", outfile)
        outfile.write('\n')
        for entry in randomly_selected_sentences_with_location:
            json.dump(entry, outfile)
            outfile.write('\n')

    txt_output_file_path = author_and_title + 'paragraph.jsonl'
    with open(txt_output_file_path, 'w', encoding='utf-8') as outfile:
        json.dump(f"Title: {title}", outfile)
        outfile.write('\n')
        json.dump(f"Author: {author}", outfile)
        outfile.write('\n')
        for entry in randomly_selected_paragraphs_with_location:
            json.dump(entry, outfile)
            outfile.write('\n')

if __name__ == "__main__":
    # Get the names of all .txt files in the current working directory:
    for file_name in os.listdir('.'):
        if file_name.endswith('.txt'):
            run(file_name)


import json
import random
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')  # Ensure the 'punkt' tokenizer is downloaded


# Look up table 
lookup_table = {
    'pg4280': 'kant_TheCritiqueofPureReason',
    'pg6763': 'Aristotle_OnTheArtOfPoetry',
    'pg2412': 'Aristotle_Categories',
    'pg59058': 'Aristotle_HistoryOfAnimals',
    'pg26095': 'Aristotle_TheAthenianConstitution',
}



# Load and parse the TXT content
txt_file_path = 'pg4280.txt'
with open(txt_file_path, 'r', encoding='utf-8') as file:
    txt_content = file.read()

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
        if len(sentence) > 10:  # Only include sentences longer than 10 characters: make the sentences meaningful and remove corporas like "Section V."
            sentences.append(sentence)
        locations.append(start_location)
        location = start_location + len(sentence)  # Update location to search for the next sentence

    # Combine sentences with their locations
    sentences_with_locations = [{"sentence": sen, "location": loc} for sen, loc in zip(sentences, locations)]
    return sentences_with_locations

# Split the extracted text into sentences with locations
all_sentences_with_location = split_into_sentences_with_location(txt_content)

# Randomly select 1000 sentences from the list, ensuring the list is large enough
randomly_selected_sentences_with_location = random.sample(all_sentences_with_location, min(1000, len(all_sentences_with_location)))

# Save the JSONL data to a new file
txt_output_file_path = 'kant_TheCritiqueofPureReason.jsonl'
with open(txt_output_file_path, 'w', encoding='utf-8') as outfile:
    for entry in randomly_selected_sentences_with_location:
        json.dump(entry, outfile)
        outfile.write('\n')

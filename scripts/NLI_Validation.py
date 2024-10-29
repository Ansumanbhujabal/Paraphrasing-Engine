import os
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
import warnings
from .Paraphrasing_test import paraphrase_pg
warnings.filterwarnings("ignore")
import nltk
nltk.download('punkt')

# Create the /log folder if it doesn't exist
log_dir = './log'
os.makedirs(log_dir, exist_ok=True)

# Set up logging configuration
logging.basicConfig(
    filename=os.path.join(log_dir, 'nli_logs.log'),
    filemode='w',  # append mode
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Load pre-trained NLI model (roberta-large-mnli)
nli_model_name = "roberta-large-mnli"
nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)

def nli_inference(premise, hypothesis):
    """Perform NLI inference and log results."""
    logging.info(f"Performing NLI inference between premise: '{premise}' and hypothesis: '{hypothesis}'")
    
    # Tokenize the inputs
    inputs = nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, padding=True)
    outputs = nli_model(**inputs)

    # Get probabilities for entailment, neutral, and contradiction
    probs = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]

    labels = ["Contradiction", "Neutral", "Entailment"]
    result_label = labels[probs.argmax()]
    
    # Log the result
    logging.info(f"NLI Label: {result_label}, Probabilities: {probs}")
    
    return result_label, probs

def split_sentences(text):
    """Use nltk to split text into sentences for better segmentation."""
    text = text.replace("\n", " ")  
    text = text.replace(". ", ".")  
    text = text.replace(".", ". ")  
    return nltk.tokenize.sent_tokenize(text)

def nli_for_corpus(original_corpus, paraphrased_corpus):
    """Process each sentence in a corpus, calculate the NLI label and probabilities, and log results."""
    original_sentences = split_sentences(original_corpus)
    paraphrased_sentences = split_sentences(paraphrased_corpus)
    
    logging.info(f"Original corpus length: {len(original_sentences)} sentences")
    logging.info(f"Paraphrased corpus length: {len(paraphrased_sentences)} sentences")

    all_probs = []  
    results = []

    # Iterate through sentences
    for original_sentence, paraphrased_sentence in zip(original_sentences, paraphrased_sentences):
        # Skip cases where either sentence is empty
        if original_sentence.strip() == "" or paraphrased_sentence.strip() == "":
            logging.warning(f"Skipping empty sentence pair: original='{original_sentence}', paraphrased='{paraphrased_sentence}'")
            continue

        # Perform NLI inference and log it
        nli_label, nli_probs = nli_inference(original_sentence, paraphrased_sentence)
        all_probs.append(nli_probs)

        # Log individual sentence processing
        logging.info(f"Processed original: '{original_sentence}', paraphrased: '{paraphrased_sentence}'")
        logging.info(f"Resulting NLI label: {nli_label}, Probabilities: {nli_probs}")

        # Append result for the current sentence
        results.append({
            'original': original_sentence,
            'paraphrased': paraphrased_sentence,
            'nli_label': nli_label,
            'nli_probabilities': nli_probs
        })

    # Calculate average probabilities
    if all_probs:
        avg_probs = np.mean(np.array(all_probs), axis=0)
        avg_label = ["Contradiction", "Neutral", "Entailment"][np.argmax(avg_probs)]
    else:
        avg_probs = np.array([0.33, 0.33, 0.33])
        avg_label = "Neutral"

    # Log average results
    logging.info(f"Average NLI label: {avg_label}, Average probabilities: {avg_probs}")
    
    return  avg_label, avg_probs

# # Example Usage
# text = """
# Option to continue as a FTE post internship completion.Join a dynamic and innovative startup environment with a supportive team.Competitive compensation package, including equity options.
# Opportunity to make a significant impact on the product and the company's growth.Work with experienced founders who have a successful track record in the industry.Perks and Benefits Company equity .
# Meals (lunch/dinner/snacks) on the house.
# Nice and comfortable office space
# """


# paraphrased_text = paraphrase_pg(text)

# # nli_results, avg_label, avg_probs = nli_for_corpus(text, paraphrased_text)
# print( nli_for_corpus(text, paraphrased_text))


import os
import warnings
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass
from time import time

import nltk
from nltk.corpus import wordnet
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers import pipeline


# Set up NLTK
nltk.download('wordnet')

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return synonyms

def generate_response(text):
    # Check if the user input contains any science-related keywords
    science_keywords = {"science", "biology", "chemistry", "physics"}
    input_words = set(text.split())
    if not input_words.intersection(science_keywords):
        return "I'm sorry, I can only answer questions about science."

    # Find the main science-related keyword in the user input
    main_keyword = None
    for word in input_words:
        if word in science_keywords:
            main_keyword = word
            break

    # Get synonyms for the main keyword
    synonyms = get_synonyms(main_keyword)

    # Generate a response using the synonyms
    response = None
    for synonym in synonyms:
        input_text = text.replace(main_keyword, synonym)
        generated_text = generator(input_text, max_length=1000, pad_token_id=tokenizer.eos_token_id)[0]['generated_text']
        if generated_text != text:
            response = generated_text
            break

    # If a response couldn't be generated using the synonyms, generate a default response
    if not response:
        response = generator("I'm sorry, I don't know the answer to that question.", max_length=1000, pad_token_id=tokenizer.eos_token_id)[0]['generated_text']

    return response.strip()
# Load the model
# model = pipeline('text-generation', model='microsoft/DialoGPT-medium', device=0)
# response = model(text, max_length=50)[0]['generated_text']

# Set the maximum length of the generated text
MAX_LENGTH = 100

############################################################
# Callback function called on update config
############################################################
def config(configuration: ConfigClass):
    # TODO Add code here
    configuration.load()
    # Set the maximum length of the generated text
    global MAX_LENGTH
    MAX_LENGTH = configuration.get("max_length", MAX_LENGTH)


############################################################
# Callback function called on each execution pass
############################################################

def execute3(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    output = []
    for text in request.text:
        response = generate_response(text)
        output.append(response)

    return SimpleText(dict(text=output))

def execute2(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    output = []
    model = pipeline('text-generation', model='microsoft/DialoGPT-medium', device=0)

    for text in request.text:        
        response = model(text, max_length=50)[0]['generated_text']
        output.append(response)

    return SimpleText(dict(text=output))


def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    #load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    # model = pipeline('text-generation', model='microsoft/DialoGPT-medium', device=0)
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    output = []
    chat_history_ids = torch.tensor([])
    print(request.text)
    for text in request.text:
        try:
            # encoding the user ids
            user_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')
            # ading users input to history
            bot_input_ids = torch.cat([chat_history_ids, user_input_ids], dim=-1) if len(chat_history_ids) > 0 else user_input_ids
            # generating response 
            chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

            output.append(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True))

        except Exception as e:
            print("Error occurred: ", e)

    return SimpleText(dict(text=output))

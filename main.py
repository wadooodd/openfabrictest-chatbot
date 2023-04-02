import os
import warnings
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass
from time import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers import pipeline

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
            bot_input_ids = torch.cat([chat_history_ids, user_input_ids], dim=-1) if len(chat_history_ids) > 0 else new_user_input_ids
            # generating response 
            chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

            output.append(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True))

        except Exception as e:
            print("Error occurred: ", e)

    return SimpleText(dict(text=output))

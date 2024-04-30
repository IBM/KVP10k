import os
import sys
from pathlib import Path

MY_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = str(Path(MY_DIR).parent)
sys.path.append(os.path.dirname(ROOT_DIR))

import transformers
from tokenizers.processors import TemplateProcessing
# import lightning.pytorch as pl

class PromptUtils():
    def __init__(self, tokenizer_type):
        print('Using mistral tokenizer')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')
            
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.bos_token = self.tokenizer.eos_token
        self.tokenizer._tokenizer.post_processor = TemplateProcessing(
            single="[bos] $A [eos]",
            special_tokens=(
                ("[bos]", self.tokenizer.bos_token_id),
                ("[eos]", self.tokenizer.eos_token_id)
            ),
        )

    def get_num_of_tokens(self, prompt):
        prompts = [prompt]

        # tokenize
        encoding = self.tokenizer(prompts,
                              max_length=None,
                              padding=True,
                              truncation=True,
                              return_tensors='pt')

        input_ids, attention_mask = encoding['input_ids'], encoding['attention_mask']
        return input_ids.shape[1]



    @staticmethod
    def make_prompt(instruction, input_, output=""):
        return "{0}\n{1}\n{2}\n{3}\n{4}\n{5}\n{6}\n{7}\n{8}\n{9}\n{10}\n{11}\n{12}".format(
            "<Document>",  # 0
            input_,  # 1
            "</Document>",  # 2
            "<Task>",  # 3
            "From the document, extract the text keys and values. ", # 4
            "Please provide the response in the form of a Python list of lists. ", #5
            "It should begin with “[“ and end with “]”", # 6
            "Each internal list should contain two comma separated string items - key and value, and should begin with “[“ and end with “]” as well. ", # 7
            "The example of format is as following:",  # 8
            "[[\"key1\":\"value1\"], [\"key2\":\"value2\"], [\"key3\":\"value3\"]]", # 9
            "</Task>",   # 10
            "### Response:",  # 11
            output,  # 12
        )


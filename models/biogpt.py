from transformers import BioGptTokenizer, BioGptForCausalLM

import torch

class biogpt(torch.nn.Module):
    def __init__(self):
        super(biogpt, self).__init__()
        self.tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
        self.model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")

    def forward(self, text):
        encoded_input = self.tokenizer(text, return_tensors='pt')
        output = self.model(**encoded_input)
        return output

    def save_pretrained(self, path):
        self.tokenizer.save_pretrained(path)
        self.model.save_pretrained(path)

    def from_pretrained(self, path):
        self.tokenizer = BioGptTokenizer.from_pretrained(path)
        self.model = BioGptForCausalLM.from_pretrained(path)



tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)

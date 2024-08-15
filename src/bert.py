import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class BERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer =  BertTokenizer.from_pretrained('bert-base-uncased')
        self.m = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, x):
        out = self.m(**x)
        return out.last_hidden_state

    def tokenize(self, texts, max_length = 64):
        return self.tokenizer(texts, max_length = max_length, padding='max_length', truncation=True, return_tensors='pt')
    
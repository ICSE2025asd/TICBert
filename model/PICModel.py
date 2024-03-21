import torch
import torch.nn as nn
from transformers import AutoModel


class PICModel(nn.Module):
    def __init__(self, pretrained_path):
        super(PICModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_path)
        self.classifier = nn.Linear(768, 1)
        self.softmax = nn.Softmax(dim=-1)
        nn.init.xavier_normal_(self.classifier.weight.data)

    def forward(self, input_ids, attention_mask, time_decay):
        x = self.encoder(input_ids=input_ids, attention_mask=attention_mask)['pooler_output']
        x = x * time_decay
        x = self.classifier(x)
        x = torch.sigmoid(x)
        return x





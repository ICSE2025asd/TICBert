import torch
import torch.nn as nn
from transformers import AutoModel


class RobertaClassifier(nn.Module):
    def __init__(self, pretrained_path, class_num):
        super(RobertaClassifier, self).__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_path)
        self.classifier = nn.Linear(768, class_num)
        nn.init.xavier_normal_(self.classifier.weight.data)

    def forward(self, input_ids, attention_mask):
        x = self.encoder(input_ids=input_ids, attention_mask=attention_mask)['pooler_output']
        x = self.classifier(x)
        x = torch.sigmoid(x)
        return x

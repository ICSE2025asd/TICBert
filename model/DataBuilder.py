import random
from enum import unique, Enum
from typing import List

import torch
from torch.utils.data import DataLoader
from FeatureEngineering.TimeWeight import time_parser

from model.TextInitializeDataset import TextInitializeDataset


class DataBuilder:

    def __init__(self, random_seed: int, division: list, tokenizer):
        self.random_seed = random_seed
        self.division = [d / sum(division) for d in division]
        self.tokenizer = tokenizer

    def collate_fn(self, batch):
        pt_batch = self.tokenizer([x[0] for x in batch], padding=True, truncation=True, max_length=64,
                                  return_tensors='pt')
        return {'input_ids': pt_batch['input_ids'],
                'attention_mask': pt_batch['attention_mask'],
                'label': torch.tensor([x[1] for x in batch]),
                'time_decay': torch.tensor([x[2] for x in batch])}

    def build(self, raw_data: List[dict], components: dict, prompt_type="", use_time_weight=False, time_weight=None):
        index, data, cur_components = 0, [], []
        raw_data.sort(key=lambda x: x["issue_key"])
        while index < len(raw_data):
            cur_components.append(raw_data[index]["component"])
            if index == len(raw_data) - 1 or raw_data[index]["issue_key"] != raw_data[index + 1]["issue_key"]:
                raw_data[index]["component"] = cur_components
                data.append(raw_data[index])
                cur_components = []
            index = index + 1
        random.seed(self.random_seed)
        random.shuffle(data)
        train_set_size = int(len(data) * self.division[0])
        valid_set_size = int(len(data) * self.division[1])
        test_set_size = int(len(data) * self.division[2])
        train, valid, test = data[:train_set_size], data[train_set_size: train_set_size + valid_set_size], data[-test_set_size:]
        if prompt_type == "":
            train, train_true_label = input_for_common_classifier(train, components)
            valid, valid_true_label = input_for_common_classifier(valid, components)
            test, test_true_label = input_for_common_classifier(test, components)
        else:
            train, train_true_label = prompt_for_PIC(train, components, prompt_type, use_time_weight, time_weight)
            valid, valid_true_label = prompt_for_PIC(valid, components, prompt_type, use_time_weight, time_weight)
            test, test_true_label = prompt_for_PIC(test, components, prompt_type, use_time_weight, time_weight)
        train_dataloader = DataLoader(TextInitializeDataset(train), batch_size=len(components.keys()), shuffle=False, collate_fn=self.collate_fn)
        valid_dataloader = DataLoader(TextInitializeDataset(valid), batch_size=128, shuffle=False, collate_fn=self.collate_fn)
        test_dataloader = DataLoader(TextInitializeDataset(test), batch_size=128, shuffle=False, collate_fn=self.collate_fn)
        return train_dataloader, train_true_label, valid_dataloader, valid_true_label, test_dataloader, test_true_label


def prompt_for_PIC(raw_data: List[dict], component_id: dict, prompt_type: str, use_time_weight: bool, time_weight):
    components, true_label, data = list(component_id.keys()), [], []
    for issue in raw_data:
        cur_components = issue["component"]
        if use_time_weight:
            create_time = issue["create_time"]
            time_decay = time_weight.get_time_weight(create_time)
        else:
            time_decay = 1
        prompts = prompt_builder_for_PIC_roberta(prompt_type, issue, components, cur_components, time_decay)
        if len(prompts) != 0:
            data.extend(prompts)
            true_label.append([component_id[_] for _ in cur_components])
    return data, true_label


def prompt_builder_for_PIC_roberta(prompt_type: str, issue: dict, components: list, cur_components: list, time_decay: float):
    prompts = []
    if prompt_type == "S":
        text = issue["text"]
        for c in components:
            if c in cur_components:
                prompts.append((f"{text}. \n This is {c} component.", 1, time_decay))
            else:
                prompts.append((f"{text}. \n This is {c} component.", 0, time_decay))
    elif prompt_type == "SCt":
        text, create_time = issue["text"], time_parser(issue["create_time"], "BY")
        if create_time is not None:
            for c in components:
                if c in cur_components:
                    prompts.append((f"{text}. This issue is created in {create_time}. \n This is {c} component.", 1, time_decay))
                else:
                    prompts.append((f"{text}. This issue is created in {create_time}. \n This is {c} component.", 0, time_decay))
    elif prompt_type == "ST":
        text, issue_type = issue["text"], issue["issue_type"]
        if issue_type is not None:
            for c in components:
                if c in cur_components:
                    prompts.append((f"{text}. This issue is a {issue_type}-type issue. \n This is {c} component.", 1, time_decay))
                else:
                    prompts.append((f"{text}. This issue is a {issue_type}-type issue. \n This is {c} component.", 0, time_decay))
    elif prompt_type == "SC":
        text, creator = issue["text"], issue["creator"]
        if creator is not None:
            for c in components:
                if c in cur_components:
                    prompts.append((f"{text}. This issue is created by {creator}. \n This is {c} component.", 1, time_decay))
                else:
                    prompts.append((f"{text}. This issue is created by {creator}. \n This is {c} component.", 0, time_decay))
    elif prompt_type == "STC":
        text, issue_type, creator = issue["text"], issue["issue_type"], issue["creator"]
        if issue_type is not None and creator is not None:
            for c in components:
                if c in cur_components:
                    prompts.append((f"{text}. This issue is a {issue_type}-type issue created by {creator}. \n This is {c} component.", 1, time_decay))
                else:
                    prompts.append((f"{text}. This issue is a {issue_type}-type issue created by {creator}. \n This is {c} component.", 0, time_decay))
    elif prompt_type == "STCt":
        text, issue_type, create_time = issue["text"], issue["issue_type"], time_parser(issue["create_time"], "BY")
        if issue_type is not None and create_time is not None:
            for c in components:
                if c in cur_components:
                    prompts.append((f"{text}. This issue is a {issue_type}-type issue created in {create_time}. \n This is {c} component.", 1, time_decay))
                else:
                    prompts.append((f"{text}. This issue is a {issue_type}-type issue created in {create_time}. \n This is {c} component.", 0, time_decay))
    elif prompt_type == "SCCt":
        text, creator, create_time = issue["text"], issue["creator"], time_parser(issue["create_time"], "BY")
        if creator is not None and create_time is not None:
            for c in components:
                if c in cur_components:
                    prompts.append((f"{text}. This issue is created by {creator} in {create_time}. \n This is {c} component.", 1, time_decay))
                else:
                    prompts.append((f"{text}. This issue is created by {creator} in {create_time}. \n This is {c} component.", 0, time_decay))
    elif prompt_type == "STCCt":
        text, issue_type, creator, create_time = issue["text"], issue["issue_type"], issue["creator"], time_parser(issue["create_time"], "BY")
        if issue_type is not None and creator is not None and create_time is not None:
            for c in components:
                if c in cur_components:
                    prompts.append((f"{text}. This issue is a {issue_type}-type issue created by {creator} in {create_time}. \n This is {c} component.", 1, time_decay))
                else:
                    prompts.append((f"{text}. This issue is a {issue_type}-type issue created by {creator} in {create_time}. \n This is {c} component.", 0, time_decay))
    elif prompt_type == "SACt":
        text, assignee, create_time = issue["text"], issue["assignee"], time_parser(issue["create_time"], "BY")
        if assignee is not None and create_time is not None:
            for c in components:
                if c in cur_components:
                    prompts.append((f"{text}. This issue is created in {create_time} and assigned to {assignee}. \n This is {c} component.", 1, time_decay))
                else:
                    prompts.append((f"{text}. This issue is created in {create_time} and assigned to {assignee}. \n This is {c} component.", 0, time_decay))
    elif prompt_type == "SA":
        text, assignee = issue["text"], issue["assignee"]
        if assignee is not None:
            for c in components:
                if c in cur_components:
                    prompts.append((f"{text}. This issue is assigned to {assignee}. \n This is {c} component.", 1, time_decay))
                else:
                    prompts.append((f"{text}. This issue is assigned to {assignee}. \n This is {c} component.", 0, time_decay))
    return prompts


def input_for_common_classifier(raw_data: List[dict], component_id: dict):
    true_label, data, class_num = [], [], len(component_id.keys())
    for issue in raw_data:
        text, cur_components, time_decay = issue["text"], issue["component"], 1
        if text is None:
            continue
        true_label_ids = [component_id[_] for _ in cur_components]
        label = [0] * class_num
        for i in true_label_ids:
            label[i] = 1
        true_label.append(true_label_ids)
        data.append((text, label, time_decay))
    return data, true_label


@unique
class NFRSO(Enum):
    Availability = 0
    Performance = 1
    Maintainability = 2
    Portability = 3
    Scalability = 4
    Security = 5
    Fault_Tolerance = 6


@unique
class PromiseNFR(Enum):
    A = "Availability"
    FT = "Fault-tolerance"
    L = "Legality"
    LF = "Look & Feel"
    MN = "Maintainability"
    O = "Operational"
    PE = "Performance"
    PO = "Portability"
    SC = "Scalability"
    SE = "Security"
    US = "Usability"


def promise_for_roberta(raw_data: list):
    data, true_label = [], []
    for line in raw_data:
        text, label = line[2].strip('"').strip(), line[3]
        index = 0
        if label == "F":
            continue
        for req in PromiseNFR:
            if label == req.name:
                data.append((f"{text} \n This is {req.value} Requirement.", 1))
                true_label.append(index)
            else:
                data.append((f"{text} \n This is {req.value} Requirement.", 0))
            index = index + 1
    return data, true_label


def nfrso_for_roberta(raw_data: list):
    data, true_label = [], []
    for line in raw_data:
        text, label = line[0], int(line[1])
        for req in NFRSO:
            if label == req.value:
                data.append((f"{text} \n This is {req.name} Requirement.", 1))
                true_label.append(label)
            else:
                data.append((f"{text} \n This is {req.name} Requirement.", 0))
    return data, true_label

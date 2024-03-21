import torch
from sklearn.metrics import recall_score, precision_score, f1_score

from util.DataLoader import summary_component_dataloader


def in_project_generalizability(model, data_builder, project: str, class_interval: list, output_k: int, cuda_device):
    for num in range(class_interval[1], class_interval[0] - 1, -1):
        data, components = summary_component_dataloader(project, num)
        _, _, _, _, test_dataloader, test_true_label = data_builder.build(data, components)
        recall, top, scale = PIC_evaluate(model, test_dataloader, test_true_label, project, num, output_k, cuda_device)
    return


def recall_k(predict_label: list, true_label: list, output_k=10):
    record = {}
    for k in range(1, output_k + 1):
        recall = 0
        for i in range(0, len(predict_label)):
            hit = set(predict_label[i][:k]) & set(true_label[i])
            recall += len(hit) / len(true_label[i])
        record[k] = recall / len(true_label)
    return record


def top_k(predict_label: list, true_label: list, output_k=10):
    record = {}
    for k in range(1, output_k + 1):
        top = 0
        for i in range(0, len(predict_label)):
            hit = set(predict_label[i][:k]) & set(true_label[i])
            if len(hit) > 0:
                top += 1
        record[k] = top / len(true_label)
    return record


def recall_k_by_label(predict_label: list, true_label: list, class_num: int, top=10):
    record = {}
    label_cnt_dict = {i: 0 for i in range(0, class_num)}
    for labels in true_label:
        for l in labels:
            label_cnt_dict[l] += 1
    print(label_cnt_dict)
    for k in range(1, top + 1):
        record[k] = {i: 0 for i in range(0, class_num)}
        for i in range(0, len(predict_label)):
            hit = set(predict_label[i][:k]) & set(true_label[i])
            for l in true_label[i]:
                record[k][l] += len(hit) / len(true_label[i])
        for i in range(0, class_num):
            if label_cnt_dict[i] == 0:
                record[k][i] = -1
            else:
                record[k][i] = record[k][i] / label_cnt_dict[i]
    return record


def PIC_evaluate(model, test_dataloader, test_true_label: list, project: str, class_num: int, output_k: int,
                 cuda_device, split_by_label=False):
    model.to(cuda_device)
    with torch.no_grad():
        model.eval()
        predicts = []
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(cuda_device)
            attention_mask = batch['attention_mask'].to(cuda_device)
            time_decay = batch['time_decay'].view(-1, 1).to(cuda_device)
            outputs = model(input_ids, attention_mask, time_decay)
            predicts.extend(outputs[:, 0].view(-1).tolist())
        predicts = torch.Tensor(predicts)
        predicts = predicts.view(len(test_true_label), -1).tolist()
        predicts = [sorted(range(len(_)), key=lambda k: _[k], reverse=True) for _ in predicts]
        recall = recall_k(predicts, test_true_label, output_k)
        top = top_k(predicts, test_true_label, output_k)
        if split_by_label:
            recall_by_label = recall_k_by_label(predicts, test_true_label, class_num, output_k)
    print(f"class={class_num}, test set size: {len(test_true_label)}")
    for i in range(1, output_k + 1):
        print(f"recall@{i}: {recall[i]}")
        print(f"top@{i}: {top[i]}")
        print()
        if split_by_label:
            print(recall_by_label[i])
    return recall, top, len(test_true_label)


def PIC_validate(model, valid_dataloader, cuda_device):
    model.to(cuda_device)
    with torch.no_grad():
        model.eval()
        predicts = []
        labels = []
        for batch in valid_dataloader:
            input_ids = batch['input_ids'].to(cuda_device)
            attention_mask = batch['attention_mask'].to(cuda_device)
            time_decay = batch['time_decay'].view(-1, 1).to(cuda_device)
            outputs = model(input_ids, attention_mask, time_decay)
            predicts.extend(list(map(lambda x: round(x, 0), outputs[:, 0].view(-1).tolist())))
            labels.extend(batch["label"])
        recall = recall_score(labels, predicts, average='macro')
        precision = precision_score(labels, predicts, average='macro')
        f1 = f1_score(labels, predicts, average='macro')
    return recall, precision, f1


def roberta_classifier_evaluate(model, test_dataloader, test_true_label: list, output_k: int, cuda_device):
    model.to(cuda_device)
    class_num = 0
    with torch.no_grad():
        model.eval()
        predicts = []
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(cuda_device)
            attention_mask = batch['attention_mask'].to(cuda_device)
            outputs = model(input_ids, attention_mask)
            predicts.extend(outputs)
        class_num = len(predicts[0])
        predicts = [sorted(range(len(_)), key=lambda k: _[k], reverse=True) for _ in predicts]
        recall = recall_k(predicts, test_true_label, output_k)
        top = top_k(predicts, test_true_label, output_k)
    print(f"class={class_num}, test set size: {len(test_true_label)}")
    for i in range(1, output_k + 1):
        print(f"recall@{i}: {recall[i]}")
        print(f"top@{i}: {top[i]}")
        print()
    return recall, top, len(test_true_label)

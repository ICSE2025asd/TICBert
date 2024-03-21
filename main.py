from FeatureEngineering.TimeWeight import GaussWeight
from model.DataBuilder import DataBuilder
from model.RobertaClassifier import RobertaClassifier
from model.evaluate import in_project_generalizability, PIC_evaluate, roberta_classifier_evaluate
from model.train import PIC_train, roberta_classifier_train

import torch
import torch.nn as nn

from transformers import AutoTokenizer

# 数据集迭代器
from model.PICModel import PICModel
from util.DataLoader import summary_component_dataloader

if __name__ == "__main__":
    # 超参设置

    mode = "train"
    model_type = "PIC"

    random_seed = 42
    division = [8, 1, 1]
    base_model = "roberta"
    pretrained_path = f'model/{base_model}'
    max_epoch_num = 36
    cuda_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    project = "ARIES"
    component_range = 25
    output_k = 10

    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    data_builder = DataBuilder(random_seed, division, tokenizer)

    if model_type == "PIC":
        use_time_weight = False
        param_type = "std"
        granularity = "timestamp"
        prompt_type = "SC"
        do_label_augmentation = False
        augment_type = "detailed"

        # 数据预处理
        data, components = summary_component_dataloader(project, component_range, do_label_augmentation, augment_type)
        component_range = min(len(components.keys()), component_range)
        if use_time_weight:
            time_series = [_["create_time"] for _ in data]
            time_weight = GaussWeight(time_series, param_type, granularity)
        else:
            time_weight = None
        train_dataloader, _, valid_dataloader, _, test_dataloader, test_true_label = \
            data_builder.build(data, components, prompt_type, use_time_weight, time_weight)
        model = PICModel(pretrained_path)
        model.to(cuda_device)

        if mode == "train":
            optimizer = torch.optim.AdamW(params=model.parameters(), lr=5e-6, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0)
            loss_func = nn.BCELoss()
            best_model = PIC_train(model, train_dataloader, valid_dataloader, max_epoch_num, optimizer, loss_func, cuda_device)
            model_name = f"{project}_{prompt_type}_"
            if use_time_weight:
                model_name = model_name + f"TW{param_type}_"
            model_name = model_name + f"{base_model}_{component_range}class_sigmoid.pkl"
            torch.save(best_model, f"model/bestmodel/{project}/{model_name}")
            model.load_state_dict(best_model)
            PIC_evaluate(model, test_dataloader, test_true_label, project, component_range, output_k, cuda_device, split_by_label=False)
        elif mode == "test":
            model_name = f"{project}_{prompt_type}_"
            if use_time_weight:
                model_name = model_name + f"TW{param_type}_"
            model_name = model_name + f"{base_model}_{component_range}class_sigmoid.pkl"
            model.load_state_dict(torch.load(f"model/bestmodel/{project}/{model_name}"))
            PIC_evaluate(model, test_dataloader, test_true_label, project, component_range, output_k, cuda_device, split_by_label=False)

    elif model_type == "roberta_classifier":
        data, components = summary_component_dataloader(project, component_range)
        component_range = min(len(components.keys()), component_range)
        train_dataloader, _, valid_dataloader, valid_true_label, test_dataloader, test_true_label = data_builder.build(data, components)
        model = RobertaClassifier(pretrained_path, component_range)
        model.to(cuda_device)
        if mode == "train":
            optimizer = torch.optim.AdamW(params=model.parameters(), lr=5e-6, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0)
            loss_func = nn.BCELoss()
            best_model = roberta_classifier_train(model, train_dataloader, valid_dataloader, valid_true_label, max_epoch_num, optimizer, loss_func, cuda_device, use_early_stop=False)
            model_name = f"{project}_{component_range}class.pkl"
            torch.save(best_model, f"model/controlgroup/{model_type}/{model_name}")
            model.load_state_dict(best_model)
            roberta_classifier_evaluate(model, test_dataloader, test_true_label, output_k, cuda_device)
        elif mode == "test":
            model_name = f"{project}_{component_range}class.pkl"
            model.load_state_dict(torch.load(f"model/controlgroup/{model_type}/{model_name}"))
            roberta_classifier_evaluate(model, test_dataloader, test_true_label, output_k, cuda_device)

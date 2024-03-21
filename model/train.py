from copy import deepcopy

from tqdm import tqdm

from model.evaluate import PIC_validate, roberta_classifier_evaluate
from util.DataLoader import summary_component_dataloader


def PIC_train(model, train_dataloader, valid_dataloader, max_epoch_num: int, optimizer, loss_func, cuda_device,
              use_early_stop=True, patience=5, min_epoch=10):
    model.to(cuda_device)
    max_valid_f1, best_model, early_stop_counter = 0, {}, 0
    for e in tqdm(range(max_epoch_num)):
        for batch in train_dataloader:
            model.train()
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(cuda_device)
            attention_mask = batch['attention_mask'].to(cuda_device)
            time_decay = batch['time_decay'].view(-1, 1).to(cuda_device)
            outputs = model(input_ids, attention_mask, time_decay)
            train_loss = loss_func(outputs[:, 0].view(-1), batch['label'].float().to(cuda_device))
            train_loss.backward()
            optimizer.step()

        recall, precision, f1 = PIC_validate(model, valid_dataloader, cuda_device)
        print(f"epoch: {e + 1}")
        print(f"recall: {recall}")
        print(f"precision: {precision}")
        print(f"f1: {f1}")
        if f1 > max_valid_f1:
            best_model = deepcopy(model.state_dict())
            max_valid_f1 = f1
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        if use_early_stop and early_stop_counter >= patience and e > min_epoch:
            print(f"Early stopping at epoch {e + 1}")
            break
    return best_model


def roberta_classifier_train(model, train_dataloader, valid_dataloader, valid_true_label, max_epoch_num: int, optimizer,
                             loss_func, cuda_device, use_early_stop=True, patience=5, min_epoch=10):
    model.to(cuda_device)
    max_valid_recall, best_model, early_stop_counter = 0, {}, 0
    for e in tqdm(range(max_epoch_num)):
        for batch in train_dataloader:
            model.train()
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(cuda_device)
            attention_mask = batch['attention_mask'].to(cuda_device)
            outputs = model(input_ids, attention_mask)
            train_loss = loss_func(outputs, batch['label'].float().to(cuda_device))
            train_loss.backward()
            optimizer.step()
        print(f"epoch: {e + 1}")
        recall, top, _ = roberta_classifier_evaluate(model, valid_dataloader, valid_true_label, 1, cuda_device)
        if recall[1] > max_valid_recall:
            best_model = deepcopy(model.state_dict())
            max_valid_recall = recall[1]
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        if use_early_stop and early_stop_counter >= patience and e > min_epoch:
            print(f"Early stopping at epoch {e + 1}")
            break
    return best_model

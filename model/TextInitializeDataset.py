from torch.utils.data import Dataset


class TextInitializeDataset(Dataset):
    def __init__(self, input_data) -> None:
        self.text = [x[0] for x in input_data]
        self.label = [x[1] for x in input_data]
        self.time_decay = [x[2] for x in input_data]

    def __getitem__(self, index):
        return [self.text[index], self.label[index], self.time_decay[index]]

    def __len__(self):
        return len(self.text)

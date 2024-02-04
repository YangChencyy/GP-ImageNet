from torch.utils.data import Dataset

class RemappedDataset(Dataset):
    def __init__(self, dataset, label_mapping):
        self.dataset = dataset
        self.label_mapping = label_mapping

    def __getitem__(self, index):
        img, label = self.dataset[index]
        # Remap the label
        remapped_label = self.label_mapping[label]
        return img, remapped_label

    def __len__(self):
        return len(self.dataset)

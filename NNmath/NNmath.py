import torch

class DataLoader():

    def __init__(self, tr_path, te_path):
        self.tr = pd.read_csv(tr_path)
        self.te = pd.read_csv(te_path)
        self.tr_idx = 0
        self.te_idx = 0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self) -> tuple:
        return (self.tr.iloc[self.tr_idx])

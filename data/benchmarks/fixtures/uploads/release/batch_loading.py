import torch
from torch.utils.data import Dataset, DataLoader
class Temperatures(Dataset):
    def __init__(self):
        self.values = [11, 13, 17, 19, 23]
    def __len__(self):
        return len(self.values)
    def __getitem__(self, index):
        return self.values[index]
loader = DataLoader(Temperatures(), batch_size=2, shuffle=False, drop_last=True, num_workers=0)
with torch.no_grad():
    baseline = torch.tensor([2.0], requires_grad=True) * 3

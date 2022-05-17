from torch.utils.data import Dataset


class MyDataset (Dataset):
    def __init__(self, X, Y):
        self.x = X
        self.y = Y
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x) 

import numpy as np
import torch

from pathlib import Path
from torch.utils.data import Dataset

class FCGRDataset(Dataset):
    def __init__(self, fcgr_dir, transform=None, target_transform=None, label2int = {"clade_G":0, "random":1}
):
        # self.img_labels = pd.read_csv(annotations_file)
        self.list_fcgr = list(Path(fcgr_dir).rglob("*.npy"))
        self.transform = transform
        self.target_transform = target_transform
        self.label2int = label2int
    def __len__(self):
        return len(self.list_fcgr)

    def __getitem__(self, idx):
        fcgr_path = self.list_fcgr[idx]
        image = np.load(fcgr_path)
        label = str(fcgr_path.parent).split("-")[-1]
        label = self.label2int[label]
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return torch.tensor(np.expand_dims(image, axis = 0),dtype=torch.float32), label
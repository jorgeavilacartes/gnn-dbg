"""
Convolutional graph neural network acting on a dBG
"""

import torch 

class CNNFCGR(torch.nn.Module):
    def __init__(self, num_classes, kmer = 6):
        super().__init__()
        self.cnn1 = torch.nn.Conv2d(in_channels=1, out_channels=4*6, kernel_size=2, stride=2, padding=0, bias=False)
        self.cnn2 = torch.nn.Conv2d(in_channels=4*6, out_channels=4*5, kernel_size=2, stride=2, padding=0, bias=False)
        self.cnn3 = torch.nn.Conv2d(in_channels=4*5, out_channels=4*4, kernel_size=2, stride=2, padding=0, bias=False)
        self.flatten = torch.nn.Flatten()
        self.dense = torch.nn.Linear(in_features=4*4*4**(kmer-3), out_features=num_classes)

    def forward(self, x):
        x = self.cnn1(x) # out_channels1 x 2**(k-1) x 2**(k-1)
        x = self.cnn2(x) # out_channels2 x 2**(k-2) x 2**(k-2)
        x = self.cnn3(x) # out_channels3 x 2**(k-3) x 2**(k-3)
        x = self.flatten(x)
        x = self.dense(x) # out_channels3 * 4**(k-3)
        return x
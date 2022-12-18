import numpy as np
from torch.utils.data import Sampler


class RSNABalanceSampler(Sampler):
    def __init__(self, dataset, ratio=8, shuffle=True):
        self.r = ratio - 1
        self.dataset = dataset
        self.shuffle = shuffle
        self.pos_index = np.where(dataset.df.cancer>0)[0]
        self.neg_index = np.where(dataset.df.cancer==0)[0]

        self.length = self.r*int(np.floor(len(self.neg_index)/self.r))

    def __iter__(self):
        pos_index = self.pos_index.copy()
        neg_index = self.neg_index.copy()

        if self.shuffle:
            np.random.shuffle(pos_index)
            np.random.shuffle(neg_index)

        neg_index = neg_index[:self.length].reshape(-1,self.r)
        pos_index = np.random.choice(pos_index, self.length//self.r).reshape(-1,1)

        index = np.concatenate([pos_index, neg_index],-1).reshape(-1)
        return iter(index)

    def __len__(self):
        return self.length
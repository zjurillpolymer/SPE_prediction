import pandas as pd
import numpy as np
from rdkit.Chem.PandasPatcher import dataframe_applymap
from torch.utils.data import Dataset, DataLoader
import torch




class SPEdataset(Dataset):
    def __init__(self,csv_path='../data/clean_train_data.csv'):
        self.df = pd.read_csv(csv_path)
        features=list(range(0,1))+list(range(2,len(self.df.columns)))
        self.features=torch.tensor(self.df.iloc[:,features].values,dtype=torch.long).unsqueeze(1)
        self.labels=torch.tensor(self.df.iloc[:,1].values,dtype=torch.long)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

dataset = SPEdataset(csv_path='../data/clean_train_data.csv')
print(dataset.features.shape)
print(dataset.labels.shape)


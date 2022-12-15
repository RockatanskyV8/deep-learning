import torch

import random
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torchvision.transforms as transforms

class Reader():
    def __init__(self, df, train_p = 0.7, val_p = 0.15, test_p = 0.15, random_seed = 42):
        self.train_p     = train_p
        self.val_p       = val_p  
        self.test_p      = test_p 
        self.df          = df 
        self.random_seed = random_seed

    def read(self):
        pd.set_option('display.max_columns', None)

        torch.manual_seed(self.random_seed)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        train_size = int(self.train_p * self.df.shape[0])
        val_size   = int(self.val_p   * self.df.shape[0])
        test_size  = int(self.test_p  * self.df.shape[0])

        X_train, X_test, y_train, y_test = train_test_split(self.df[['x', 'y', 'z']],
                                                    self.df['label'],
                                                    train_size=train_size,
                                                    stratify=self.df['label'],
                                                    random_state=42)

        X_valid, X_test, y_valid, y_test = train_test_split(X_test,
                                                    y_test,
                                                    test_size=test_size,
                                                    stratify=y_test,
                                                    random_state=42)

        X_train.reset_index(drop=True, inplace=True)
        X_valid.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True)

        ss = StandardScaler()
        ss.fit(X_train)

        X_train = ss.transform(X_train)
        X_valid = ss.transform(X_valid)
        X_test = ss.transform(X_test)
        
        transformations = transforms.Compose([
                            transforms.ToTensor()
                          ])

        y_train = pd.get_dummies(y_train, prefix='target').reset_index(drop=True)
        y_valid = pd.get_dummies(y_valid, prefix='target').reset_index(drop=True)
        y_test  = pd.get_dummies(y_test, prefix='target').reset_index(drop=True)

        return [X_train, X_test, y_train, y_test], [X_valid, X_test, y_valid, y_test]

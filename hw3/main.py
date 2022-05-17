import pandas
import sklearn.preprocessing, sklearn.decomposition
from sklearn.model_selection import train_test_split
from sklearn.utils import column_or_1d
from sklearn_pandas import DataFrameMapper
import numpy as np

import torch
import torch.nn as nn
import torch.functional as F
import torch.optim


def pre_process_data():
    pri_star_table = pandas.read_csv (
        "/mnt/2022spr/data-integrate/data/pri_star_info_202205141729.csv"
    )
    cust_asset = pandas.read_csv(
        "/mnt/2022spr/data-integrate/data/pri_cust_asset_info_202205141729.csv",
        usecols=[
            "uid",
            "all_bal",
            "avg_mth",
            "avg_year",
            "sa_bal",
            "td_bal",
            "fin_bal",
            "sa_crd_bal",
            "sa_td_bal",
            "ntc_bal",
            "td_3m_bal",
            "cd_bal",
            "etl_dt"
        ]
    )

    res = pandas.merge(pri_star_table, cust_asset, how='left', on='uid')
    res = res[(res['star_level']!=-1)]
    res = res[(res['all_bal'].notnull())]

    train_mapper = DataFrameMapper ([
        (['all_bal'], sklearn.preprocessing.StandardScaler()),
        (['avg_mth'], sklearn.preprocessing.StandardScaler()),
        (['avg_year'], sklearn.preprocessing.StandardScaler()),
        (['sa_bal'], sklearn.preprocessing.StandardScaler()),
        (['sa_crd_bal'], sklearn.preprocessing.StandardScaler()),
    ])

    X = np.round (train_mapper.fit_transform(res.copy()), 2)

    labels_mapper = DataFrameMapper([
        (['star_level'], None)
    ])

    Y = np.round (labels_mapper.fit_transform(res.copy()))

    X_train, X_test, Y_train, Y_test = train_test_split (
        X, Y, test_size=0.2
    )   
    
    Y_train = column_or_1d(Y_train)
    Y_test = column_or_1d(Y_test)

    tensor_X_train = torch.from_numpy(X_train).float()
    tensor_Y_train = torch.from_numpy(Y_train)

    return tensor_X_train, tensor_Y_train


class MyNetwork(nn.Module):
    def __init__(self) :
        super(MyNetwork, self).__init__()

        #self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential (
            nn.Linear(5, 20),
            nn.SELU(),
            nn.Linear(20, 9),
            nn.SELU()
        )
    def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train_loop(X, Y, model, loss_fn, optimizer):

    print(X.dtype, Y.dtype)

    pred = model(X)
    loss = loss_fn(pred, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss = loss.item()
    print(f"loss: {loss:>7f}")


if __name__ == "__main__":
    model = MyNetwork()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-3)
    trainX, trainY = pre_process_data()
    
    train_loop(trainX, trainY, model, loss_fn, optimizer)



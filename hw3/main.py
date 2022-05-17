import pandas
import sklearn.preprocessing, sklearn.decomposition
from sklearn.model_selection import train_test_split
from sklearn.utils import column_or_1d
from sklearn_pandas import DataFrameMapper
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
import numpy as np

import torch
import torch.nn as nn
import torch.functional as F
import torch.optim
from torch.utils.data import DataLoader

from mydata import MyDataset


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
            "etl_dt",
            "sa_td_bal"
        ]
    )

    cust_asset_acct = pandas.read_csv (
        "/mnt/2022spr/data-integrate/data/pri_cust_asset_acct_info_202205141729.csv",
        usecols=[
            "uid",
            "subject_no",   #OrdinalEncoder
            "prod_type",    #same
            "acct_char",    #same
            "frz_sts",      #same
            "stp_sts",      #same
            "acct_bal"     #standardScaler
        ]
    )

#    pri_cust_base = pandas.read_csv (
#        "/mnt/2022spr/data-integrate/data/pri_cust_base_info_202205141729.csv",
#        usecols=[
#            'uid',
#            'marrige',
#            'education',
#            'career',
#            'prof_titl',
#            'is_employee',
#            'is_shareholder',
#            'is_black'
#        ]
#    )

    # merge 1
    res = pandas.merge(pri_star_table, cust_asset, how='left', on='uid')
    res = res[(res['star_level']!=-1)]
    res = res[(res['all_bal'].notnull())]
    
    # merge 2
    res = pandas.merge(res, cust_asset_acct, how="left", on="uid")
    res = res[(res['subject_no'].notnull())]

#    # merge 3
#    res = pandas.merge(res, pri_cust_base, how='left', on='uid')
#    res = res[(res['marrige'].notnull())]

    train_mapper = DataFrameMapper ([
        (['all_bal'], sklearn.preprocessing.StandardScaler()),
        (['avg_mth'], sklearn.preprocessing.StandardScaler()),
        (['avg_year'], sklearn.preprocessing.StandardScaler()),
        (['sa_bal'], sklearn.preprocessing.StandardScaler()),
        (['td_bal'], sklearn.preprocessing.StandardScaler()),
        (['fin_bal'], sklearn.preprocessing.StandardScaler()),
        (['sa_crd_bal'], sklearn.preprocessing.StandardScaler()),
        (['acct_bal'], sklearn.preprocessing.StandardScaler()),

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

    tensor_X_test = torch.from_numpy(X_test).float()
    tensor_Y_test = torch.from_numpy(Y_test)

    return tensor_X_train, tensor_Y_train, tensor_X_test, tensor_Y_test


class MyNetwork(nn.Module):
    def __init__(self) :
        super(MyNetwork, self).__init__()

        #self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential (
            nn.Linear(8, 20),
            nn.SELU(),
            nn.Linear(20, 10),
            nn.SELU(),
        )
    def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train_loop(dataloader, model, loss_fn, optimizer):

    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss

        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == "__main__":

    train_x, train_y, test_x, test_y = pre_process_data()

    clf = DecisionTreeClassifier()

    clf.fit(train_x, train_y)
    pred = clf.predict(test_x)
    print(confusion_matrix(test_y, pred))
    
    #train_data = MyDataset(train_x, train_y)
    #test_data = MyDataset(test_x, test_y)
#
    #train_data_loader = DataLoader (
    #    train_data, batch_size=64
    #)
    #test_data_loader = DataLoader (
    #    test_data, batch_size=64
    #)
#
    #model = MyNetwork()
    #loss_fn = nn.CrossEntropyLoss()
    #learning_rate = 1e-3
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
    #epochs = 10
    #for t in range(epochs):
    #    print(f"Epoch {t+1}\n-------------------------------")
    #    train_loop(train_data_loader, model, loss_fn, optimizer)
    #    test_loop(test_data_loader, model, loss_fn)
    #print("Done!")

  
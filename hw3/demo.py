from cmath import nan
import pandas
import sklearn.preprocessing, sklearn.decomposition
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import column_or_1d
from sklearn_pandas import DataFrameMapper
import numpy as np

import torch
import torch.nn as nn

from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB, BernoulliNB, CategoricalNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV

def show_cust_base():
    cust_base = pandas.read_csv(
        "/mnt/2022spr/data-integrate/data/pri_cust_base_info_202205141729.csv", 
        usecols=[
            "uid",
            "marrige",
            "education",
            "career",
            "prof_titl",
            "is_employee",
            "is_shareholder",
            "is_black"       
        ]
    )

    classification = "is_black"

    print(
        cust_base[["uid", classification]].groupby(classification).count()
    )

def show_cust_asset():
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
    classification = "etl_dt"
    print(
        cust_asset[["uid", classification]].groupby(classification).count()
    )

def show_cust_asset_acct():
    cust_asset_acct = pandas.read_csv(
        "/mnt/2022spr/data-integrate/data/pri_cust_asset_acct_info_202205141729.csv",
        usecols=[
            "uid",
            "subject_no",
            "prod_type",
            "acct_char",
            "deps_type",
            "prod_code",
            "is_secu_card",
            "acct_sts",
            "frz_sts",
            "stp_sts",
            "acct_bal",
            "bal",
            "avg_mth",
            "avg_year",
            "etl_dt"
        ]
    )
    classification = "etl_dt"
    print(
        cust_asset_acct[["uid", classification]].groupby(classification).count()
    )
    
def show_cust_liab_acct():
    cust_liab_acct = pandas.read_csv(
        "/mnt/2022spr/data-integrate/data/pri_cust_liab_acct_info_202205141729.csv",
        usecols=[
            "uid",
            "loan_type",
            "loan_amt",
            "loan_bal",
            "vouch_type",
            "is_mortgage",
            "five_class",
            "overdue_flag",
            "owed_int_flag",
            "credit_amt",
            "delay_bal",
            "guar_amount",
            "guar_con_value"
        ]
    )
    classification = "guar_con_value"
    print(
        cust_liab_acct[["uid", classification]].groupby(classification).count()
    )

def show_pri_star_info():
    pri_star_info = pandas.read_csv(
        "/mnt/2022spr/data-integrate/data/pri_star_info_202205141729.csv",
    )
    print (pri_star_info[['uid', 'star_level']].groupby('star_level').count())

def show_pri_credit_info():
    pri_credit_info = pandas.read_csv(
        "/mnt/2022spr/data-integrate/data/pri_credit_info_202205141729.csv",
    )
    print (pri_credit_info)

def some_preprocess():
    # 准备数据
    print ("Prepare the data")

    data = pandas.DataFrame(
        {
            'pet':      ['cat', 'dog', 'fish', 'dog', 'cat'],
            'children': [3, 4, 6, 7, 9],
            'salary'  : [10, 20, 30, 40, 50]
        }
    )
    
    data['flag'] = 0
    
    print(data)
    
    mapper = DataFrameMapper ([
        (['pet'], sklearn.preprocessing.OrdinalEncoder()),
        (['children'], sklearn.preprocessing.StandardScaler()),
        (['salary'], sklearn.preprocessing.StandardScaler()),
        (['children', 'salary'], sklearn.decomposition.PCA(1))
    ], df_out=True)
    
    post_data = np.round (mapper.fit_transform(data.copy()), 2)
    print(post_data)

def merge_table():
    data1 = pandas.DataFrame (
        {
            'key' : [1, 2, 3, 4],
            'salary' : [2., 3., 4., 5.1]
        }
    )

    data2 = pandas.DataFrame (
        {
            'key' : [4, 3, 2, 1],
            'months' : [5, 2, 1, 3]
        }
    )

    res = pandas.merge(data1, data2, how='left', on='key')

    print(res)

if __name__ == "__main__":
    show_pri_star_info()


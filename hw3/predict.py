import pandas
from sklearn_pandas import DataFrameMapper
import sklearn.preprocessing
import numpy as np
import joblib

def star_predict():
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

    # merge 1
    res = pandas.merge(pri_star_table, cust_asset, how='left', on='uid')
    res = res[(res['all_bal'].notnull())]
    
    # merge 2
    res = pandas.merge(res, cust_asset_acct, how="left", on="uid")
    res = res[(res['subject_no'].notnull())]

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
    clf = joblib.load("../model/star_decision_tree.joblib")
    labels = clf.predict(X)

    for rowIdx, label in enumerate(labels):
        if res.iloc[rowIdx]['star_level']==-1:
            res.loc[res.index[rowIdx], 'star_level'] = label
        if rowIdx%1000==0:
            print("{:.4f}".format(rowIdx/len(labels)))
    res = res[['uid','star_level']]
    res.to_csv('star_predict.csv',index=False)

def credit_predict():
    pri_cust_liab = pandas.read_csv(
        "/mnt/2022spr/data-integrate/data/pri_cust_liab_info_202205141729.csv",
        usecols=[
            'uid',
            'all_bal',
            'bad_bal',
            'due_intr',
            'norm_bal',
            'delay_bal',
        ]
    )

    pri_credit_info = pandas.read_csv(
        "/mnt/2022spr/data-integrate/data/pri_credit_info_202205141729.csv"
    )

    pri_credit_info['credit_level'] = pri_credit_info['credit_level'].map(
        lambda x : 
        1 if x==35 else 
            (2 if x==50 else (
                3 if x==60 else (
                    4 if x==70 else (
                        5 if x==85 else -1
                    )
                )
            ))
    )

    res = pandas.merge(pri_credit_info, pri_cust_liab, how='left', on='uid')
    res = res[(res['all_bal'].notnull())]

    dm_v_as_djk = pandas.read_csv(
        "/mnt/2022spr/data-integrate/data/dm_v_as_djk_info_202205141729.csv",
        usecols=[
            'uid',
            'cred_limit',
            'over_draft',
            'dlay_amt'
        ]
    )

    res = pandas.merge(res, dm_v_as_djk, how='left', on='uid')
    res = res[(res['cred_limit'].notnull())]

    # 不使用下面这张表，因为连接后数据量将从接近20w下降至1w条，因此不使用
    #dm_v_as_djkfq = pandas.read_csv (
    #    "/mnt/2022spr/data-integrate/data/dm_v_as_djkfq_info_202205141729.csv",
    #    usecols=[
    #        'uid',
    #        'total_amt'
    #    ]
    #)
    #res = pandas.merge(res ,dm_v_as_djkfq, how='left', on='uid')
    #res = res[(res['total_amt'].notnull())]

    train_mapper = DataFrameMapper ([
        (['all_bal'], sklearn.preprocessing.StandardScaler()),
        (['bad_bal'], sklearn.preprocessing.StandardScaler()),
        (['due_intr'], sklearn.preprocessing.StandardScaler()),
        (['norm_bal'], sklearn.preprocessing.StandardScaler()),
        (['delay_bal'], sklearn.preprocessing.StandardScaler()),
        (['cred_limit'], sklearn.preprocessing.StandardScaler()),
        (['over_draft'], sklearn.preprocessing.StandardScaler()),
        (['dlay_amt'], sklearn.preprocessing.StandardScaler())
    ])

    X = np.round(train_mapper.fit_transform(res.copy()), 2)

    clf = joblib.load("../model/credit_decision_tree.joblib")
    labels = clf.predict(X)

    convert = lambda x : 35 if x==1 else (50 if x==2 else ( 60 if x==3 else (70 if x==4 else (85 if x==5 else -1))))

    for rowIdx, label in enumerate(labels):
        if res.iloc[rowIdx]['credit_level']==-1:
            res.loc[res.index[rowIdx], 'credit_level'] = convert(label)
        else:
            res.loc[res.index[rowIdx], 'credit_level'] = convert(res.loc[res.index[rowIdx], 'credit_level'])

        if rowIdx%1000==0:
            print("{:.4f}".format(rowIdx/len(labels)))
    res = res[['uid','credit_level']]
    res.to_csv('credit_predict.csv',index=False)


if __name__=="__main__":
    credit_predict()
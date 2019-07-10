# -*- coding: utf-8 -*-
# @Time    : 2019/6/12 上午11:36
# @Author  : Lin_QH

import pandas as pd
import numpy as np
import os
from scipy import sparse
from scipy.stats import kurtosis
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()



def  train_csv(path):

    train_set = pd.read_csv(path,parse_dates=['auditing_date', 'due_date', 'repay_date'])
    train_set['repay_date'] = train_set[['due_date', 'repay_date']].apply(
        lambda x: x['repay_date'] if x['repay_date'] != '\\N' else x['due_date'], axis=1
    )
    train_set['repay_amt'] = train_set['repay_amt'].apply(lambda x: x if x != '\\N' else 0).astype('float32')
    train_set['label'] = (train_set['repay_date'] - train_set['auditing_date']).dt.days
    train_set.loc[train_set['repay_amt'] == 0, 'label'] = 32

    return  train_set

def merge_features(train_set,features,id=None):

    new_train = train_set.merge(features,how="left",on=[id])

    return new_train

def user_features(u_features,name):

    new_features = u_features.sort_values(by="insertdate",ascending=False).drop_duplicates("user_id").reset_index(drop=True)
    new_features.rename(columns={"insertdate":name},inplace=True)

    return new_features

def add_user_info_features(train):
    print("==========add_user_info_features==========")
    # 3. 借款用户基础信息表（user_info.csv）
    user_info = pd.read_csv("../data_analysis/dataset/user_info.csv",parse_dates=['reg_mon', 'insertdate'])

    user_info['reg_mon'] = lb.fit_transform(user_info['reg_mon'])
    user_info['gender'] = lb.fit_transform(user_info['gender'])
    user_info['cell_province'] = lb.fit_transform(user_info['cell_province'])
    user_info['id_province'] = lb.fit_transform(user_info['id_province'])
    user_info['id_city'] = lb.fit_transform(user_info['id_city'])

    user_info.sort_values(by=['user_id', 'insertdate'], inplace=True)
    user_info.rename(columns={"insertdate":'info_insert_date'},inplace=True)
    a1 = user_info.ix[user_info.groupby(['user_id'])['info_insert_date'].tail(1).index, :]
    a2 = user_info.groupby(['user_id'], as_index=False)['gender'].agg({'user_info_count':'count'})

    for a in [a1,a2]:
        train = pd.merge(train, a, on=['user_id'], how='left')
    return train

def add_user_taglist_features(train):
    print("==========add_user_taglist_features==========")
    # 4. 用户画像标签列表（user_taglist.csv）
    user_taglist = pd.read_csv("../data_analysis/dataset/user_taglist.csv",parse_dates=["insertdate"])

    a1 = user_taglist.groupby(['user_id'], as_index=False)['insertdate'].agg({'user_insertdate_count': 'count'})
    user_taglist = pd.merge(user_taglist,a1,on=['user_id'],how="left")
    user_taglist.sort_values(by=['user_id', 'insertdate'], inplace=True)
    user_taglist = user_taglist.ix[user_taglist.groupby(['user_id'])['insertdate'].tail(1).index, :]

    user_taglist['taglist'] = user_taglist['taglist'].map(lambda x: x.split('|'))
    user_taglist['taglist_len'] = user_taglist['taglist'].map(lambda x: len(x))
    user_taglist.rename(columns={"insertdate":'tag_insert_date'},inplace=True)

    # if os.path.exists( './user_taglist_word2vec.model'):
    #     tag_model = Word2Vec.load('./user_taglist_word2vec.model')
    # else:
    #     tag_model = word2vec_fit(user_taglist['taglist'], user_taglist['taglist_len'].max())
    #     tag_model.save('user_taglist_word2vec.model')
    train = pd.merge(train,user_taglist,on=['user_id'],how='left')

    return train

def add_listing_info_features(train):
    print("==========add_listing_info_features==========")
    # 2.标的属性表（listing_info.csv）
    listing_info = pd.read_csv("../data_analysis/dataset/listing_info.csv")
    listing_info['principal/term'] = listing_info['principal'] / listing_info['term']

    train = pd.merge(train, listing_info[['listing_id', 'term', 'rate', 'principal']],
                     on=['listing_id'], how='left')
    a1 = listing_info.groupby(['user_id'], as_index=False)['term'].agg(
        {'user_listing_count': 'count', 'user_term_mean': 'mean',
         'user_term_min': 'min', 'user_term_max': 'max', 'user_term_sum': 'sum'})

    a2 = listing_info.groupby(['user_id'], as_index=False)['rate'].agg(
        {'user_rate_mean': 'mean','user_rate_min': 'min', 'user_rate_max': 'max', 'user_rate_var': 'var'})

    a3 = listing_info.groupby(['user_id'], as_index=False)['principal'].agg(
        {'user_principal_mean': 'mean', 'user_principal_min': 'min', 'user_principal_max': 'max',
         'user_principal_var': 'var', 'user_principal_sum': 'sum'})

    a4 = listing_info.groupby(['user_id'], as_index=False)['principal/term'].agg(
        {'user_principal/term_mean': 'mean', 'user_principal/term_min': 'min', 'user_principal/term_max': 'max',
         'user_principal/term_var': 'var', 'user_principal/term_sum': 'sum'})
    del listing_info['user_id'], listing_info['auditing_date']

    for a in [a1, a2, a3, a4]:
        train = pd.merge(train, a, on=['user_id'], how='left')

    return train
def gen_repay_gap_day(x):
    if x[1] == '2200-01-01':
        return np.nan
    else:
        return (pd.to_datetime(x[0]) - pd.to_datetime(x[1])).days
def add_user_repay_features(train):

    print("==========add_user_repay_features==========")
    # 6.用户还款日志表（user_repay_logs.csv）
    user_repay_logs = pd.read_csv("../data_analysis/dataset/user_repay_logs.csv",parse_dates=['due_date', 'repay_date'])

    user_repay_logs['overdue'] = user_repay_logs['repay_date'].astype('str').apply(lambda x: 1 if x != '2200-01-01' else 0)

    user_repay_logs['gap_date'] = user_repay_logs[['due_date', 'repay_date']].apply(gen_repay_gap_day, axis = 1)
#     user_repay_logs['gap_date'] = (user_repay_logs["due_date"] - user_repay_logs["repay_date"]).dt.days

    a1 = user_repay_logs.groupby(['user_id'], as_index=False)['listing_id'].agg({
        'user_repay_count':'count','user_repay_nunique':'nunique'})
    a2 = user_repay_logs.groupby(['user_id'], as_index=False)['order_id'].agg(
        {'user_repay_order_id_mean': 'mean'})
    a3 = user_repay_logs.groupby(['user_id'], as_index=False)['due_amt'].agg(
        {'user_repay_due_amt_mean': 'mean', 'user_repay_due_amt_sum': 'sum', 'user_repay_due_amt_max': 'max'})

    a4 = user_repay_logs.groupby(['user_id'], as_index=False)['overdue'].agg(
        {'user_id_overdue_rate': 'mean'})

    a5 = user_repay_logs.groupby(['user_id'], as_index=False)['gap_date'].agg(
        {'user_repay_gap_date_mean': 'mean', 'user_repay_gap_date_sum': 'sum', 'user_repay_gap_date_max': 'max', 'user_repay_gap_date_var': 'var'})

    for a in [a1, a2, a3, a4, a5]:
        #去掉重复的user
        a = a.drop_duplicates("user_id").reset_index(drop=True)
        train = pd.merge(train, a, on=['user_id'], how='left')
    return train

# def urlog_features(user_repay_logs,order_id=1):
#
#     urlogs = user_repay_logs[user_repay_logs["order_id"] == order_id].reset_index(drop=True)
#     urlogs['repay'] = urlogs['repay_date'].astype('str').apply(
#         lambda x: 1 if x != '2200-01-01' else 0)
#     urlogs["repay_days"] = (urlogs["due_date"] - urlogs["repay_date"]).dt.days
#     urlogs.loc[urlogs["repay"] == 0, "repay_days"] = 32
#     new_urlog = urlogs[["user_id", "repay", "repay_days", "due_amt"]]
#     group = new_urlog.groupby("user_id", as_index=False)
#     new_urlog = new_urlog.merge(group["repay"].agg({"repay_mean": "mean"}), how="left", on=["user_id"])
#     new_urlog = new_urlog.merge(
#         group["repay_days"].agg({"repay_day_max": "max", "repay_days_sum": "sum", "repay_days_median": "median",
#                                  "repay_days_mean": "mean", "repay_days_std": "std"}), how="left", on=["user_id"])
#     new_urlog = new_urlog.merge(
#         group['due_amt'].agg({
#             'due_amt_max': 'max', 'due_amt_min': 'min', 'due_amt_median': 'median',
#             'due_amt_mean': 'mean', 'due_amt_sum': 'sum', 'due_amt_std': 'std',
#             'due_amt_skew': 'skew', 'due_amt_kurt': kurtosis, 'due_amt_ptp': np.ptp
#         }), on='user_id', how='left')
#     del new_urlog["repay"], new_urlog["repay_days"], new_urlog["due_amt"]
#     new_urlog = new_urlog.drop_duplicates("user_id").reset_index(drop=True)
#
#     return new_urlog

def add_user_behavior_features(train):
    print("==========add_user_behavior_features==========")
    # 5.借款用户操作行为日志表（user_behavior_logs.csv）
    user_behavior_logs = pd.read_csv("../data_analysis/dataset/user_behavior_logs.csv")

    a1 = user_behavior_logs.groupby(['user_id'], as_index=False)['behavior_type'].agg({'user_behavior_count':'count'})
    a2 = user_behavior_logs.groupby(['user_id', 'behavior_type']).size().unstack().reset_index()
    a2.columns = ['user_id'] + ['behavior_type_' + str(i) for i in a2.columns[1:]]

    for a in [a1, a2]:

        train = pd.merge(train, a, on=['user_id'], how='left')

    return train


def date_type_process(train_set,date_cols):
    for i in date_cols:
        if i in date_cols[2:-1]:
            train_set[i + "_year"] = train_set[i].dt.year
        train_set[i + "_month"] = train_set[i].dt.month
        if i in date_cols[0:3]:
            train_set[i + "_day"] = train_set[i].dt.day
            train_set[i + "_dayofweek"] = train_set[i].dt.dayofweek
    train_set.drop(columns=date_cols, axis=1, inplace=True)
    return train_set


if __name__ == "__main__" :

    train_df = train_csv("../data_analysis/dataset/train.csv")

    df1 = train_df["label"]
    print(df1)
    df1.to_csv("./date_labels.csv", index=None)
    df2 = train_df["repay_amt"]
    df2.to_csv("./amt_labels.csv", index=None)

    del train_df['repay_date'], train_df["label"], train_df["repay_amt"]

    test_df = pd.read_csv('../data_analysis/dataset/test.csv', parse_dates=['auditing_date', 'due_date'])

    df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    # 1.listing_info
    df = add_listing_info_features(df)
    # 2.user_info

    df = add_user_info_features(df)

    df = add_user_taglist_features(df)

    df = add_user_repay_features(df)

    df = add_user_behavior_features(df)

    df['due_amt_per_days'] = df['due_amt'] / (train_df['due_date'] - train_df['auditing_date']).dt.days
    date_cols = ['auditing_date', 'due_date', 'reg_mon', 'info_insert_date', 'tag_insert_date']
    for date in date_cols:
        df[date] = pd.to_datetime(pd.Series(df[date]))

    new_df = date_type_process(df, date_cols)

    print(new_df)
    df[:1000000].to_csv("./new_train_set.csv", index=None)
    df[1000000:].to_csv("./new_test_set.csv", index=None)
    print(df[1000000:].shape)
    print(df.shape)
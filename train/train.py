# -*- coding: utf-8 -*-
# @Time    : 2019/6/13 下午8:24
# @Author  : Lin_QH

from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from scipy import sparse
import pandas as pd
import numpy as np


if __name__ == "__main__":


    #数据集路径
    train_path = "../data_preprocess/new_train_set.csv"
    test_path = "../data_preprocess/new_test_set.csv"
    label1_path = '../data_preprocess/date_labels.csv'
    label2_path = '../data_preprocess/amt_labels.csv'
#以下为test小样
    #读入数据集
    # columns = pd.read_csv(train_path,nrows=0).columns.drop('taglist')
    path = "../data_analysis/dataset/test.csv"
    test_df = pd.read_csv(path)
    sub = test_df[['listing_id', 'auditing_date', 'due_amt']]
    train_set = pd.read_csv(train_path,nrows=1000)
    x_train_due_amt = train_set[["due_amt"]]
    test_set = pd.read_csv(test_path,nrows=1000)
    label_days = pd.read_csv(label1_path,nrows=1000,header=None)
    label_amt = pd.read_csv(label2_path,nrows=1000,header=None)
    print(type(train_set))
    print(type(test_set))

    train_num = train_set.shape[0]
    # 处理taglist
    cv = CountVectorizer(min_df=10, max_df=0.9)
    tag_cv1 = cv.fit_transform(train_set['taglist'].astype("str"))
    del train_set['taglist'],train_set['user_id'],train_set['listing_id']

    tag_cv2 = cv.transform(test_set['taglist'].astype("str"))
    del test_set['taglist'],test_set['user_id'],test_set['listing_id']
    # train_set = train_set._df()
    # print(train_set)
# nan处理
    train_set = train_set.fillna(axis=1,method="ffill")
    test_set = test_set.fillna(axis=1,method="ffill")
    print(train_set)

    new_train_set = sparse.hstack((train_set.values, tag_cv1), format='csr', dtype="float32")
    new_test_set = sparse.hstack((test_set.values, tag_cv2), format='csr', dtype="float32")
    print(type(new_train_set))
    print(type(new_test_set))

    # 特征选择

    # X = new_train_set
    # y = label_days.values.reshape(-1,1)
    # clf = ExtraTreesClassifier()
    # clf = clf.fit(X, y)
    # print("features_importance:",clf.feature_importances_)
    # model = SelectFromModel(clf, prefit=True)
    # X_new = model.transform(X).toarray()
    # print("新数据:",X_new)
    # print('新标签',y)
    # print(X_new.shape)
    # print(new_train_set.shape)



    # 划分数据集
    x_train, x_test, y_train= new_train_set,new_test_set,label_days.values
    print(x_train)
    print(x_test)


    # 标准化处理
    std = StandardScaler(with_mean=False)
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    lgb = LGBMClassifier(learning_rate=0.5,
                         n_estimators=10,
                         subsample=0.8,
                         subsample_freq=1,
                         colsample_bytree=0.8)
    grd = GradientBoostingClassifier(max_depth=8,n_estimators=10,learning_rate=0.5)
    ada = AdaBoostClassifier(learning_rate=0.5,n_estimators=10)
    sv = SVC(probability=True)

    clfs = list([grd, ada, sv])

    n_folds = 3
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    l = list(skf.split(x_train, y_train))
    train_sets = np.zeros((x_train.shape[0], len(clfs)))
    test_sets = np.zeros((x_test.shape[0], len(clfs)))
    for j, clf in enumerate(clfs):
        '''依次训练各个单模型'''
        print(j, clf)
        test_j = np.zeros((x_test.shape[0], len(l)))
        for i, (trn_idx, val_idx) in enumerate(skf.split(x_train, y_train)):
            '''使用第i个部分作为预测，剩余的部分来训练模型，获得其预测的输出作为第i部分的新特征。'''
            # print("Fold", i)
            trn_x, trn_y = x_train[trn_idx], y_train[trn_idx]
            val_x, val_y = x_train[val_idx], y_train[val_idx]
            clf.fit(trn_x, trn_y)
            y_submission = clf.predict(val_x)[:, 1]
            train_sets[val_idx, j] = y_submission
            test_j[:, i] = clf.predict(x_test)[:, 1]
        '''对于测试集，直接用这k个模型的预测值均值作为新的特征。'''
        test_sets[:, j] = test_j.mean(axis=1)

    print(train_sets)
    print(test_sets)

    lgb.fit(train_sets, y_train)
    result = lgb.predict(test_sets)[:, 1]
    result = (result-result.min())/(result.max()-result.min())
    print(result)
    print(result.shape)





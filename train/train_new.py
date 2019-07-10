# -*- coding: utf-8 -*-
# @Time    : 2019/6/13 下午8:24
# @Author  : Lin_QH

from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.externals import joblib
from scipy import sparse
import pandas as pd
import numpy as np

if __name__ == "__main__":


    # 数据集路径
    train_path = "../data_preprocess/new_train_set.csv"
    test_path = "../data_preprocess/new_test_set.csv"
    label1_path = '../data_preprocess/date_labels.csv'
    label2_path = '../data_preprocess/amt_labels.csv'
    # 读入数据集
    path = "../data_analysis/dataset/test.csv"
    test_df = pd.DataFrame(pd.read_csv(path, parse_dates=['auditing_date']))
    sub = test_df[['listing_id', 'auditing_date', 'due_amt']]

    train_set = pd.DataFrame(pd.read_csv(train_path))

    x_train_due_amt = train_set[["due_amt"]]
    test_set = pd.read_csv(test_path)

    label_days = pd.read_csv(label1_path, names=['label_days'])

    label_days = pd.DataFrame(label_days)['label_days'].values

    label_amt = pd.read_csv(label2_path, names=['label_amt'])

    label_amt = pd.DataFrame(label_amt)['label_amt'].values

    train_num = train_set.shape[0]

    # 处理taglist
    cv = CountVectorizer(min_df=10, max_df=0.9)
    tag_cv1 = cv.fit_transform(train_set['taglist'].astype("str"))
    del train_set['taglist'],train_set['user_id'],train_set['listing_id']

    tag_cv2 = cv.transform(test_set['taglist'].astype("str"))
    del test_set['taglist'],test_set['user_id'],test_set['listing_id']


    new_train_set = sparse.hstack((train_set.values, tag_cv1), format='csr', dtype="float32")
    new_test_set = sparse.hstack((test_set.values, tag_cv2), format='csr', dtype="float32")


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
    x_train, x_test, y_train= new_train_set,new_test_set,label_days
    # print(x_train)
    # print(x_test)

    # 标准化处理\可使用
    # std = StandardScaler(with_mean=False)
    # x_train = std.fit_transform(x_train)
    # x_test = std.transform(x_test)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
    clf = LGBMClassifier(learning_rate=0.1,
                         n_estimators=300,
                         subsample=0.8,
                         subsample_freq=1,
                         colsample_bytree=0.8)
    # grd = GradientBoostingClassifier(max_depth=8,n_estimators=10,learning_rate=0.5)
    # ada = AdaBoostClassifier(learning_rate=0.5,n_estimators=10)
    #
    # train_sets = []
    # test_sets = []
    # for clf in [grd, ada,lgb]:
    #     train_set, test_set = get_stacking(clf,x_train, y_train, x_test)
    #     train_sets.append(train_set)
    #     test_sets.append(test_set)
    # print(train_sets)
    # print(test_sets)

    amt_oof = np.zeros(train_num)
    prob_oof = np.zeros((train_num, 33))
    test_pred_prob = np.zeros((x_test.shape[0], 33))
    for i, (trn_idx, val_idx) in enumerate(skf.split(x_train, y_train)):
        print(i, 'fold...')

        trn_x, trn_y = x_train[trn_idx], y_train[trn_idx]
        val_x, val_y = x_train[val_idx], y_train[val_idx]
        val_repay_amt = label_amt[val_idx]

        val_due_amt = x_train_due_amt.iloc[val_idx]

        clf.fit(
            trn_x, trn_y,
            eval_set=[(trn_x, trn_y), (val_x, val_y)],
            early_stopping_rounds=100, verbose=5
        )
        joblib.dump(clf, '../model/lgb.pkl')
        # shape = (-1, 33)
        val_pred_prob_everyday = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)
        prob_oof[val_idx] = val_pred_prob_everyday
        val_pred_prob_today = [val_pred_prob_everyday[i][val_y[i]] for i in range(val_pred_prob_everyday.shape[0])]
        val_pred_repay_amt = val_due_amt['due_amt'].values * val_pred_prob_today
        print('val rmse:', np.sqrt(mean_squared_error(val_repay_amt, val_pred_repay_amt)))
        print('val mae:', mean_absolute_error(val_repay_amt, val_pred_repay_amt))
        amt_oof[val_idx] = val_pred_repay_amt
        test_pred_prob += clf.predict_proba(x_test, num_iteration=clf.best_iteration_) / skf.n_splits

    print('\ncv rmse:', np.sqrt(mean_squared_error(label_amt, amt_oof)))
    print('cv mae:', mean_absolute_error(label_amt, amt_oof))
    print('cv logloss:', log_loss(label_days, prob_oof))
    print('cv acc:', accuracy_score(label_days, np.argmax(prob_oof, axis=1)))

    prob_cols = ['prob_{}'.format(i) for i in range(33)]
    for i, f in enumerate(prob_cols):
        sub[f] = test_pred_prob[:, i]

    sub_example = pd.read_csv('../data_analysis/dataset/submission.csv', parse_dates=['repay_date'])
    sub_example = sub_example.merge(sub, on='listing_id', how='left')
    print(sub_example)
    sub_example['days'] = (sub_example['repay_date'] - sub_example['auditing_date']).dt.days
    # shape = (-1, 33)
    test_prob = sub_example[prob_cols].values
    test_labels = sub_example['days'].values

    test_prob = [test_prob[i][test_labels[i]] for i in range(test_prob.shape[0])]
    sub_example['repay_amt'] = sub_example['due_amt'] * test_prob
    sub_example[['listing_id', 'repay_date', 'repay_amt']].to_csv('./new_submission.csv', index=False)


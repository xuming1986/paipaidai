# paipaidai
可参阅data_analysis文件了解比赛及数据

model模型acc_score在0.40885，2折交叉验证，使用lightgbm训练，参数设置简单，估计器参数（n_estimator）设为20，学习率（learning_rate）0.5
# 使用步骤
1.可自行去拍拍贷官网下载数据集，更改feature_fetch.py中文件路径即可获取新的features和labels（提取新的特征，这里融合了所有数据表，得到63个特征）  

2.train.py 文件中将lightgbm参数做了优化，训练预测acc在0.51085，有较大提升，直接运行train.py，可以得到预测结果new_submission及训练的分类模型lgb.pkl   
3.优化：考虑加入新的组合特征，融合lightgbm和XGBoost模型集成训练，可以得到更加准确的分类模型


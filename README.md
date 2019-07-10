# paipaidai
可参阅data_analysis文件了解比赛及数据

model模型acc_score在0.40885，2折交叉验证，使用lightgbm训练，参数设置简单，估计器参数（n_estimator）设为20，学习率（learning_rate）0.5
# 使用步骤
1.可自行去拍拍贷官网下载数据集，更改feature_fetch.py中文件路径即可获取新的features和labels（提取新的特征，这里融合了所有数据表，得到63个特征）
2.

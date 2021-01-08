# Q:我给你元气森林商品的一些特征(总金额、数量、品类、规格、购买时间）,你如何预测用户是否会流失?如何设计特征。如何做接下来的事情?

## 查看分析数据(让我们深入理解用户的购物行为,从个体和群体层面上)

简单看看整个数据的各方面指标.

查看数据类型.

看下是否有残缺值.

看下每用户的订单数.

客户总交易额的分布。

用户购买具体某件商品数量的分布

客户总交易额的分布。

### 观测用户的购物周期模式

一周内不同日期的商品销量.

一天内不同时段的商品销量.

用户的购物频次:查看与上次下单间隔的时间得出结论.

 

### 影响回购的因素

统计下不同商品被reorder的次数，并看下重购次数最多的前10名产品.

 

## 数据清洗

填充缺失值,去除缺失值过多的用户行.

过滤掉购买次数少于某个值的用户,比如5次.

 

 

## 构造数据集:

构造三个数据集，分别是训练集、验证集(调参,评估模型)和测试集(获得最终该模型的真实泛化能力效果。

 

以下是一个简单的小例子,可以再优化:

训练集可以设置为1/2的用户 在 1.1-6.1之前的购买行为. 预测的tag就是这些用户在接下来的1-2个月是否还会有消费行为.   [更进一步的说,我们可以预测用户U对哪些品类和规格的商品有购买,在什么时间购买,不过这个问题稍微复杂些,我暂时没有考虑.  我还想到了可以通过对和该用户类似的其它用户的用户行为来给该用户推荐他可能同时也喜欢的商品,类似于关联分析,用A-Priori算法.]

验证集为1/4的用户数据,测试集也为1/4用户数据.

 

## 特征选择

给的表其实是用户行为数据.

 

对于类别变量:

1.用“One-hot”编码创建新列，表明原始数据中每个可能值的存在(或不存在)。

2.简单的标签编码

 

1.新特征构造:

购买时间可以构造新的用户平均购买商品价值频率。

用户不活跃月份数量是不是好用的特征？

规格可以归一化,或者离散化为几个tag,比如100ml为1,500ml为2,1000ml为3.

 

 

特征选择使用随机森林,输出特征重要度排名.

 

 

## 模型选择

通过验证集来查看此类中各个算法的表现，从而选择合适的模型。

常用且效果最好的只有lightgbm和xgboost，直接用这两个算法。可以查一下sklearn.model_selection包的相关文档。

还可以用工业界比较常用的LR试试。评价指标可以是accuracy，可以是观察混淆矩阵，或者是ROC曲线和AUC的值。

 

## 参数选择

可以尝试调整下面的这些参数.

```python
parameters = {

   'max_depth': [5, 10, 15, 20, 25],

   'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],

  'n_estimators': [500, 1000, 2000, 3000, 5000],

    'min_child_weight': [0, 2, 5, 10, 20],

    'max_delta_step': [0, 0.2, 0.6, 1, 2],

    'subsample': [0.6, 0.7, 0.8, 0.85, 0.95],

    'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],

    'reg_alpha': [0, 0.25, 0.5, 0.75, 1],

    'reg_lambda': [0.2, 0.4, 0.6, 0.8, 1],

    'scale_pos_weight': [0.2, 0.4, 0.6, 0.8, 1, 8, n]

 

}

 

xlf = xgb.XGBClassifier(max_depth=10,

                       learning_rate=0.01,

                       n_estimators=2000,

                    silent=True,

                    objective='binary:logistic',

                    nthread=12,

                     gamma=0,

                      min_child_weight=1,

                       max_delta_step=0,

                       subsample=0.85,

                      colsample_bytree=0.7,

                     colsample_bylevel=1,

                        reg_alpha=0,

                       reg_lambda=1,

                      scale_pos_weight=1,

                      seed=1440,

                      missing=None)

 

gsearch = GridSearchCV(xlf, param_grid=parameters, scoring='accuracy', cv=3)

gsearch.fit(X_train, y_train)

print("Best score: %0.3f" % gsearch.best_score_)

print("Best parameters set:")

best_parameters = gsearch.best_estimator_.get_params()

for param_name in sorted(parameters.keys()):

     print("\t%s: %r" % (param_name, best_parameters[param_name]))
```

最笨的方法是网格搜索,我之前试过贝叶斯优化,会比这个好一些.

## 模型训练与评估

见模型选择部分。

 

## 模型融合

训练多个模型，然后按照一定的方法整合多个模型输出的结果，常用的有加权平均、投票、学习法等.

可以对lightgbm和xgboost两个算法预测出的概率进行简单的加权求和.

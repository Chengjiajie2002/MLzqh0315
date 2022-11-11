import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')
train_df = pd.read_csv(r'dataTrain.csv',dtype={'f6': 'category',})#导入数据
#print(train_df)
train_df['f6'] = pd.factorize(train_df.f6, sort=True)[0]
#这段代码让f3的值变成了数字值（三种），便于我们计算，low是1，mid是2，high是0
test_df = pd.read_csv(r'dataA.csv',dtype={'f6': 'category',})
test_df['f6'] = pd.factorize(test_df.f6, sort=True)[0]
#print(train_df)
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

def get_model():
    return DecisionTreeClassifier(class_weight=[1],max_depth=3)#决策树深度为3

X = train_df.values[:, 1:-1]
print(X)
#X是所有训练集中的f1-f46的值
y = train_df.values[:, -1]
print(y)
#y是所有训练集中label值
t=0
kf = KFold(10)
for train_index, test_index in kf.split(X):
    ##print(train_index)
    #train_index是行数，在excle中显示的行数-2，所以应该是0开始算
    trainX, trainY = X[train_index], y[train_index]
    #train_x是行数对应的x值，也就是f1-f46的值，train_y就是对应的label值
    ##print(trainX,trainY)
    testX, testY = X[test_index], y[test_index]#此处导入参数
    model=get_model()
    model.fit(trainX,trainY)
    print(model.score(testX, testY))
    t=model.score(testX, testY)+t
testX = test_df.values[:, 1:]
print(testX)
#print(testX)
pred = model.predict(testX)#预测结果
pred_df = pd.DataFrame(data={'id': test_df.id,'label': pred,})
pred_df.to_csv('submission.csv', index=False)
print('AUC=',t/10)
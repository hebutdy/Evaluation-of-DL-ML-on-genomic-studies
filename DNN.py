import numpy as np
import pandas as pd
import keras
import os
from keras import models
from keras import layers
from keras.layers import Dropout, BatchNormalization
from matplotlib import pyplot
from matplotlib.ticker import MultipleLocator
from numpy import column_stack
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
import sklearn.metrics
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from keras import backend as K
from sklearn.model_selection import KFold, StratifiedKFold
from keras import regularizers

os.environ["CUDA_VISIBLE_DEVICES"] = "1"   # GTX 3060

# data1 = pd.read_csv("E:/DNN预测/train.csv")
train = pd.read_csv("D:/Gene/data//10%train.csv").fillna(0)
train_X = pd.concat([train.loc[:, train.columns[1:6]], train.loc[:, train.columns[9:2009]]], axis=1) # 要训练的数据
train_X[0:].fillna(0, inplace=True)
#print(train_X)AsthmaStatus  CancerStatus   COPDStatus
#train_X = train_X.values.reshape(-1, 2008)  # reshape(2,8) #以2行8列的形式显示
#print(train_data)
train_Y = train.loc[:, 'COPDStatus']  # 要训练的数据
train_Y[0:].fillna(0, inplace=True)
print(train_Y)
#train_Y = train_Y.values.reshape(-1, 1)  # reshape(2,8) #以2行8列的形式显示
#print(train_targets)

# data2 = pd.read_csv("E:/DNN预测/test.csv")
test = pd.read_csv("D:/Gene/data/10%test.csv").fillna(0)
test_X = pd.concat([test.loc[:, test.columns[1:6]], test.loc[:, test.columns[9:2009]]], axis=1) # 要训练的数据
test_X[0:].fillna(0, inplace=True)
print(test_X)

test_Y = test.loc[:, 'COPDStatus']
test_Y[0:].fillna(0, inplace=True)
#test_Y = test_Y.values.reshape(-1, 1)



# 一个epoch 表示： 所有的数据送入网络中， 完成了一次前向计算 + 反向传播的过程
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_X.shape[1],), kernel_regularizer=regularizers.l1(0.001)))  # 传入了输入数据的预期形状
    model.add(layers.Dense(64, activation='relu'))  # 带有relu激活的全连接层（Dense）的简单堆叠Dense(64,activation='relu') 传入Dense层的参数16是该层 的隐藏单元个数，一个隐藏单元（hidden unit）是该层表示空间的一个维度。
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1(0.001)))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # 用rmsprop作为优化器   metrics衡量指标
    return model

# 训练模型
thresholds = np.arange(0, 1, 0.001)
max = 0
t = 0
num_epochs = 120
model = build_model()

history = model.fit(train_X, train_Y, validation_data=(test_X, test_Y), epochs=num_epochs, batch_size=128, verbose=1)
predict = model.predict(test_X)
print(predict)
#s = pd.DataFrame(predict)
#s.to_csv('D:\\data\\10%predict.csv', encoding='utf-8', index=False, header=False)
prey_y = predict.copy()
for threshold_index, threshold in enumerate(thresholds):
    prey_y[predict >= threshold] = 1
    prey_y[predict < threshold] = 0
    F1 = f1_score(test_Y, prey_y)
    if F1 > max:
        max = F1
        t = threshold
print('最佳阈值: ', t, '  ', '最佳F1值: ', max)

prey_y1 = predict.copy()
prey_y1[predict >= t] = 1
prey_y1[predict < t] = 0
print(prey_y1)

#精确度: precision，正确预测为正的，占全部预测为正的比例，TP/(TP+FP)
#召回率: recall，正确预测为正的，占全部实际为正的比例,TP/ (TP+FN)
#F1-score:精确率和召回率的调和平均数，2* precision*recall / (precision+recall)

#y_pred = np.round(predict)
# s = pd.DataFrame(test_Y)
# s.to_excel('D:\\基因课题\\Data\\90%对比原样本.xls', encoding='utf-8', index=False, header=False)
precision = precision_score(test_Y, prey_y1)
recall = recall_score(test_Y, prey_y1)
f1 = f1_score(test_Y, prey_y1)
auc_score = roc_auc_score(test_Y, prey_y1)
#print(f1_score(test_Y, prey_y1))
print('1. The precision of the model = ', '%0.4f' % precision)
print('2. The recall of the model = ', '%0.4f' % recall)
print('3. The f1_score of the model = ', '%0.4f' % f1)
print('4. The AUC of the model = ', '%0.4f' % auc_score)
print('5. Confusion Metrix: \n \n', confusion_matrix(test_Y, prey_y1))
print('6. The accuracy_score of the model {}\t', accuracy_score(test_Y, prey_y1))
print('7. The classification of the model {}\n', classification_report(test_Y, prey_y1))


"""
fpr, tpr, threshold = roc_curve(test_Y, prey_y1)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,6))
plt.title('Validation ROC')
plt.plot(fpr, tpr, 'b', label = 'Val AUC = %0.4f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
print(model.summary())


pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.xlabel('Epochs', fontsize = 12)
pyplot.ylabel('Loss', fontsize = 12)
pyplot.show()
"""
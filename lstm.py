import pandas as pd
from matplotlib.ticker import MultipleLocator
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
import numpy as np
from sklearn.metrics import auc, precision_score, recall_score, f1_score, accuracy_score, classification_report, \
    confusion_matrix
from numpy import  column_stack
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from tensorflow import optimizers
from tensorflow.python.keras import Sequential, regularizers
from matplotlib import pyplot
import os
from tensorflow.python.keras.layers import LSTM, Dense, Dropout, BatchNormalization
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# ---- 数据导入 ----
pd.read_csv
str1='./data'
nummmm1 = '1'
nummmm2='1'
str2='/train'
str4='/test'
str3='0%.csv'
datatrain = pd.read_csv(str1+nummmm1+str2+nummmm2+str3)
datatest = pd.read_csv(str1+nummmm1+str4+nummmm2+str3)
datatrain[0:].fillna(0, inplace=True)
datatest[0:].fillna(0, inplace=True)
train_y = datatrain.iloc[:, 6:7].values
test_y = datatest.iloc[:, 6:7].values
datatrain = datatrain.drop(['AsthmaStatus','COPDStatus','CancerStatus'], axis=1)
datatest = datatest.drop(['AsthmaStatus','COPDStatus','CancerStatus'], axis=1)
train_x = datatrain.iloc[:, :].values  # 所有行2以后的所有列
test_x = datatest.iloc[:, :].values  # 所有行2以后的所有列
scaler = preprocessing.StandardScaler().fit(test_x)
test_x = scaler.transform(test_x)
scaler = preprocessing.StandardScaler().fit(train_x)
train_x = scaler.transform(train_x)
# ---- 参数定义----
# split_point=int(len(origin_data_x)*0.75)
input_size = 2006
time_step = 1
labels = 1
batch_size = 128
train_x = train_x.reshape([-1, input_size, time_step])
print(train_x.shape)
train_x = np.transpose(train_x, [0, 2, 1])
print(train_y)
print(train_y.shape)
print(train_x.shape)
# 测试集数据
test_x = test_x.reshape([-1, input_size, time_step])
test_x = np.transpose(test_x, [0, 2, 1])
print("Data processing is finished!")
# design network
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
acc_per_fold = []
loss_per_fold = []
thresholds_foldtrain = []
thresholds_foldtest = []
fold_no = 1
def evaluating(trainY, Y_PRE,fold2):
    p, r, thresholds = precision_recall_curve(trainY, Y_PRE)
    f1_scores = (2 * p * r) / (p + r)
    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])
    pyplot.plot(thresholds, p[:-1], color='blue')
    pyplot.plot(thresholds, r[:-1], color='orange')
    pyplot.plot(thresholds, f1_scores[:-1], color='green')
    pyplot.xlabel('thresholds')
    pyplot.ylabel('pred')
    pyplot.legend(["precision", "recall","f1_score"])  # legend 在轴上设置一个图例
    strtitle2 = '100%AsthmaStatusPred:' + str(fold2) + 'th'
    pyplot.title(strtitle2)
    pyplot.show()
    Y_PRE[Y_PRE >= thresholds[best_f1_score_index]] = 1
    Y_PRE[Y_PRE < thresholds[best_f1_score_index]] = 0
    print("最佳F1：", best_f1_score, "最佳阈值", thresholds[best_f1_score_index])
    print('1. The precision of the model {}\t', precision_score(trainY, Y_PRE))
    print('2. The recall of the model {}\t', recall_score(trainY, Y_PRE))
    print('3. The f1 of the model {}\t', f1_score(trainY, Y_PRE))
    print('6. Confusion Metrix: \n \n', confusion_matrix(trainY, Y_PRE))
    return thresholds[best_f1_score_index]

def evaluating2(trainY, Y_PRE, best):
    Y_PRE[Y_PRE >= best] = 1
    Y_PRE[Y_PRE < best] = 0
    print('1. The precision of the model {}\t', precision_score(trainY, Y_PRE))
    print('2. The recall of the model {}\t', recall_score(trainY, Y_PRE))
    print('3. The f1 of the model {}\t', f1_score(trainY, Y_PRE))
    print('6. Confusion Metrix: \n \n', confusion_matrix(trainY, Y_PRE))

def mscatter(x, y, ax=None,m=None,**kw):
    import matplotlib.markers as mmarkers
    if not ax: ax = pyplot.gca()
    sc=ax.scatter(x, y, **kw)
    if (m is not None) and (len(m) == len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc

def paixu(predictvalue, spline,realvalue):
    data=column_stack((predictvalue,realvalue))
    predictvalue.sort()
    data1=pd.DataFrame(data,columns=('prob','actual'))
    alldata=data1.sort_values(by='prob')
    x=np.arange(0,len(predictvalue))
    y=alldata.loc[:,'prob']
    c=alldata.loc[:,'actual']
    m={0:'.',1:'*'}
    map_color={0:'b',1:'r'}
    map_size={0:60,1:30}
    cm=list(map(lambda z:m[z],c))
    color=list(map(lambda x:map_color[x],c))
    size = list(map(lambda x: map_size[x], c))

    fig,ax =pyplot.subplots(figsize=(30,15))
    scatter=mscatter(x,y,c=color,m=cm,ax=ax,cmap='RdYlBu',s=size)
    x_major_locator = MultipleLocator(10000)
    y_major_locator = MultipleLocator(0.1)
    ax = pyplot.gca()
    # aX为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    pyplot.xlim([0, len(predictvalue)])
    pyplot.ylim([0, 1])
    for i in range(len(predictvalue)):
        if predictvalue[i]==spline:
            p3=pyplot.vlines(i, 0, len(x), colors="green", linestyles="dashed")
            break
        if predictvalue[i]>spline:
            p3=pyplot.vlines(i, 0, len(x), colors="green", linestyles="dashed")
            break
    p4=pyplot.hlines(spline, 0, len(x), colors="red", linestyles="dashed")
    pyplot.legend([p3, p4],["vertical_line", "best_threshold"])

    pyplot.xlabel('num')
    pyplot.ylabel('pred')
    pyplot.title('100%AsthmaStatusPred')
    pyplot.show()

def paixu1(predictvalue, spline,realvalue,fold):
    data=column_stack((predictvalue,realvalue))
    predictvalue.sort()
    data1=pd.DataFrame(data,columns=('prob','actual'))
    alldata=data1.sort_values(by='prob')
    x=np.arange(0,len(predictvalue))
    y=alldata.loc[:,'prob']
    c=alldata.loc[:,'actual']
    m={0:'.',1:'*'}
    map_color={0:'b',1:'r'}
    map_size={0:60,1:30}
    cm=list(map(lambda z:m[z],c))
    color=list(map(lambda x:map_color[x],c))
    size = list(map(lambda x: map_size[x], c))

    fig,ax =pyplot.subplots(figsize=(30,15))
    scatter=mscatter(x,y,c=color,m=cm,ax=ax,cmap='RdYlBu',s=size)
    x_major_locator = MultipleLocator(10000)
    y_major_locator = MultipleLocator(0.1)
    ax = pyplot.gca()
    # aX为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    pyplot.xlim([0, len(predictvalue)])
    pyplot.ylim([0, 1])
    for i in range(len(predictvalue)):
        if predictvalue[i]==spline:
            p3=pyplot.vlines(i, 0, len(x), colors="green", linestyles="dashed")
            break
        if predictvalue[i]>spline:
            p3=pyplot.vlines(i, 0, len(x), colors="green", linestyles="dashed")
            break
    p4=pyplot.hlines(spline, 0, len(x), colors="red", linestyles="dashed")
    pyplot.legend([p3, p4],["vertical_line", "best_threshold"])
    pyplot.xlabel('num')
    pyplot.ylabel('pred')
    strtitle = '100%AsthmaStatus:' + str(fold) + 'th'
    pyplot.title(strtitle)
    pyplot.show()

def buildmodle():
    model = Sequential()
    # model.add(LSTM(30, input_shape=(train_x.shape[1], train_x.shape[2]),kernel_regularizer=regularizers.l2(0.001),activity_regularizer=regularizers.l1(0.001)))
    model.add(LSTM(64, input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=True))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(LSTM(32))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid', activity_regularizer=regularizers.l2(0.005)))
    sgd = optimizers.SGD(lr=0.0005, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=sgd)
    print(model.summary())
    return model
for train, test in kfold.split(train_x, train_y):
    model = buildmodle()
    print('---------------------------------', fold_no, '-------------------------------------')
    history = model.fit(train_x[train], train_y[train], epochs=100, batch_size=batch_size,
                        validation_data=(train_x[test], train_y[test]), verbose=1, shuffle=True)
    y_ktrain = model.predict_proba(train_x[train])
    print('训练集结果')
    besttrain = evaluating(train_y[train], y_ktrain,fold_no)
    y_ktrain2 = list(y_ktrain)
    paixu1(y_ktrain2, besttrain,train_y[train],fold_no)
    print('===============besttrain:', besttrain, '=====================')
    y_ktest = model.predict_proba(train_x[test])
    print('------------------------------', fold_no, '测试集结果--------------------------------', )
    evaluating2(train_y[test], y_ktest,besttrain)
    # test the model
    score = model.evaluate(train_x[test], train_y[test], batch_size=72, verbose=0)  # evaluate函数按batch计算在某些输入数据上模型的误差
    print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
    acc_per_fold.append(score[1] * 100)
    loss_per_fold.append(score[0])
    thresholds_foldtrain.append(besttrain)
    fold_no += 1
print("%.2f%% (+/- %.2f%%)acc" % (np.mean(acc_per_fold), np.std(acc_per_fold)))
print("%.2f%% (+/- %.2f%%)loss" % (np.mean(loss_per_fold), np.std(loss_per_fold)))
print("%.2f%% (+/- %.2f%%)thresholds" % (np.mean(thresholds_foldtrain,dtype=np.float32), np.std(thresholds_foldtrain)))
thresholds_fold1=np.mean(thresholds_foldtrain,dtype=np.float32)
print('a',thresholds_fold1)
y_probs_train = model.predict_proba(train_x)
y_probs_test = model.predict_proba(test_x)


s1=pd.DataFrame(y_probs_test,columns=['pred'])
s2=pd.DataFrame(test_y,columns=['label'])
df =pd.concat([s1,s2],axis=1)
df.to_csv('./lstmno/y_probs_testAS'+nummmm2+'0%'+nummmm1+'.csv',encoding ='utf-8',index=True, header=True)

fpr1, tpr1, thre1 = metrics.roc_curve(train_y, y_probs_train)
auc1 = auc(fpr1, tpr1)
pyplot.plot(fpr1, tpr1, 'b', label='auc=%0.2f' % auc1)
pyplot.legend(loc='lower right')
pyplot.plot([0, 1], [0, 1], 'r--')
pyplot.xlim([-0.1, 1.1])
pyplot.ylim([-0.1, 1.1])
pyplot.xlabel('False Positive Rate')  # 横坐标是fpr
pyplot.ylabel('True Positive Rate')  # 纵坐标是tpr
pyplot.title('train auc 100%AsthmaStatusReceiver operating characteristic example')
pyplot.show()
y_probs_train2 = list(y_probs_train)
paixu(y_probs_train2, np.mean(thresholds_foldtrain), train_y)
y_probs_train[y_probs_train >= thresholds_fold1] = 1
y_probs_train[y_probs_train < thresholds_fold1] = 0

print('训练集结果', )
print('auc. The precision of the model {}\t', auc1)
print('1. The precision of the model {}\t', precision_score(train_y, y_probs_train))
print('2. The recall of the model {}\t', recall_score(train_y, y_probs_train))
print('3. The f1 of the model {}\t', f1_score(train_y, y_probs_train))
print('6. Confusion Metrix: \n \n', confusion_matrix(train_y, y_probs_train))
p, r, thresholds = precision_recall_curve(train_y, y_probs_train)
f1_scores = (2 * p * r) / (p + r)
best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])
pyplot.plot(thresholds, p[:-1], color='blue')
pyplot.plot(thresholds, r[:-1], color='orange')
pyplot.plot(thresholds, f1_scores[:-1], color='green')
pyplot.xlabel('sample')
pyplot.ylabel('pred')
pyplot.legend(["precision", "recall","f1_score"])  # legend 在轴上设置一个图例
pyplot.title('100%AsthmaStatusPredtrain-----')
pyplot.show()

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.xlabel('Epochs', fontsize=12)
pyplot.ylabel('Loss', fontsize=12)
pyplot.savefig("./images/Loss_label.png")
pyplot.title('100%AsthmaStatusLOSS')
pyplot.show()

fpr1, tpr1, thre1 = metrics.roc_curve(test_y, y_probs_test)
auc1 = auc(fpr1, tpr1)
pyplot.plot(fpr1, tpr1, 'b', label='auc=%0.2f' % auc1)
pyplot.legend(loc='lower right')
pyplot.plot([0, 1], [0, 1], 'r--')
pyplot.xlim([-0.1, 1.1])
pyplot.ylim([-0.1, 1.1])
pyplot.xlabel('False Positive Rate')  # 横坐标是fpr
pyplot.ylabel('True Positive Rate')  # 纵坐标是tpr
pyplot.title('TEST auc100%AsthmaStatus Receiver operating characteristic example')
pyplot.show()
p, r, thresholds = precision_recall_curve(test_y, y_probs_test)
f1_scores = (2 * p * r) / (p + r)
best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])
pyplot.plot(thresholds, p[:-1], color='blue')
pyplot.plot(thresholds, r[:-1], color='orange')
pyplot.plot(thresholds, f1_scores[:-1], color='green')
pyplot.xlabel('sample')
pyplot.ylabel('pred')
pyplot.legend(["precision", "recall","f1_score"])  # legend 在轴上设置一个图例
pyplot.title('-----100%AsthmaStatusPredtest-----')
pyplot.show()
y_probs_test2 = list(y_probs_test)
paixu(y_probs_test2, np.mean(thresholds_foldtrain),test_y)
y_probs_test[y_probs_test >= thresholds_fold1] = 1
y_probs_test[y_probs_test < thresholds_fold1] = 0

score_all=[]
print('测试集结果')
precision=precision_score(test_y, y_probs_test)
score_all.append(precision)
recall=recall_score(test_y, y_probs_test)
score_all.append(recall)
f1=f1_score(test_y, y_probs_test)
score_all.append(f1)
score_all.append(auc1)
score_all.append(thresholds_fold1)
print(score_all)


if os.path.exists( './lstmno/score_all'+nummmm2+'0%AS.csv' ) == False:
 print('不存在')
 sa=pd.DataFrame(score_all,columns=['1'])
 index_ = ['precision', 'recall', 'f1', 'auc','thresholds']
 sa.index = index_
 sa.to_csv('./lstmno/score_all'+nummmm2+'0%AS.csv', encoding='utf-8', index=True, header=True)
else:
 print('存在')
 sa=pd.DataFrame(score_all)
 csv_data = pd.read_csv('./lstmno/score_all'+nummmm2+'0%AS.csv', low_memory = False)#example.csv是需要被追加的CSV文件，low_memory防止弹出警告
 csv_df =pd.DataFrame(csv_data)
 csv_df[nummmm1] =sa
 csv_df.to_csv('./lstmno/score_all'+nummmm2+'0%AS.csv',index = None)

print('auc. The precision of the model {}\t', auc1)
print('1. The precision of the model {}\t',precision)
print('2. The recall of the model {}\t',recall )
print('3. The f1 of the model {}\t', f1)
print('6. Confusion Metrix: \n \n', confusion_matrix(test_y, y_probs_test))

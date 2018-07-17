#数据以文书为单位进行加载，以文书为单位进行预测，预测一次输出准确度、召回率、然后再将所有文书的准确率、召回率求平均后，再求F值
#step0 gather data ws（excel format）
#step1 feed with ws
#ste2  predict
#step 4 count result
import os
import tensorflow as tf
import numpy as np
import random
import pandas as pd

from code.process.preprocess128 import preprocess
# from cnn_word2vec.predict import CnnModel


testdatapath = '事实到法条/validate-data/D-validate-20.txt'
# save_dir = '事实到法条/checkpoints/cnn-使用新训练数据-128-data465'
# save_path = os.path.join(save_dir, 'best_validation')
# cnn_model = CnnModel(save_path)


# def test():
#     #加载模型
#     word2vecpath = '2014model_size128.model'
#     p = preprocess(word2vecpath)
#     p.load_models()
#
#     precision = []
#     recall = []
#     #加载数据
#     with open(testdatapath,'r',encoding='utf-8') as f:
#         content = f.read().split('.xml_ft2jl.xls')
#         for i in range(1,len(content)):
#             data_1 = []
#             data_2 = []
#             input_y = []
#
#             item = content[i]
#             allsample = item.split('\n')[:-1]#摈弃最后一个
#             array_index = []
#
#             #先过滤样本，再进行样本向量化
#             for i in range(len(allsample)):
#                 sample = allsample[i]
#                 label = sample.split('|')[2]
#                 if label[0] == '0':
#                     array_index.append(i)
#
#
#
#             ##不过滤负例子
#             filer_index = []
#
#             # # 根据0的个数，随机删除一半0
#             # base = random.sample([i for i in range(len(array_index))], int(len(array_index) / 2))
#             # filer_index = []
#             # for index in base:
#             #     filer_index.append(array_index[index])
#
#             new_samples = []
#             for i in range(len(allsample)):
#                 if i not in filer_index:
#                     sample = allsample[i]
#                     new_samples.append(sample)
#                     text1 = sample.split('|')[0].split(' ')
#                     text2 = sample.split('|')[1].split(' ')
#                     label = sample.split('|')[2]
#
#                     if label[0] == '1':
#                         input_y.append(1)
#                     elif label[0] == '0':
#                         input_y.append(0)
#
#                     # 数据预处理
#                     data_1.append(p.fixedvec([p.vector(ss) for ss in text1], 30))
#                     data_2.append(p.fixedvec([p.vector(ft) for ft in text2], 50))
#
#             print(len(filer_index))
#             print(len(new_samples))
#             #预测
#
#             pp = 0
#             pt = 0
#             t = 0
#             y_pre = list(cnn_model.predict(data_1,data_2))
#
#
#             for y,y_,sample in zip(input_y,y_pre,new_samples):
#                 print(y,y_,sample)
#                 if y_ == 1:
#                     pp += 1
#                     if y == 1:
#                         pt += 1
#                 if y == 1:
#                     t += 1
#             if pp == 0 or t == 0:
#                 continue
#             precision_i = pt/pp
#             recall_i = pt/t
#             precision.append(precision_i)
#             recall.append(recall_i)
#             print('pp,pt,t:',pp,pt,t)
#             print('precision,recall',precision_i,recall_i)
#
#
#     print(len(precision),len(recall))
#     precision_average = np.mean(np.array(precision))
#     recall_average = np.mean(np.array(recall))
#     f = precision_average*recall_average*2/(precision_average+recall_average)
#     print(precision_average,recall_average,f)

def evaluatews(y_pre_cls,y_test_cls):
    precision = []
    recall = []
    # 加载数据
    with open(testdatapath, 'r', encoding='utf-8') as f:
        content = f.read().split('.xml_ft2jl.xls')
        base = 0
        for i in range(1, len(content)):
            pp,pt,t = 0,0,0
            item = content[i]
            allsample = item.split('\n')[:-1]  # 摈弃最后一个
            for y,y_,sample in zip(y_test_cls[base:base+len(allsample)],y_pre_cls[base:base+len(allsample)],allsample):
                print(y,y_,sample)
                if y_ == 1:
                    pp += 1
                    if y == 1:
                        pt += 1
                if y == 1:
                    t += 1
            if pp == 0 or t == 0:
                continue
            precision_i = pt/pp
            recall_i = pt/t
            precision.append(precision_i)
            recall.append(recall_i)
            print('pp,pt,t:',pp,pt,t)
            print('precision,recall',precision_i,recall_i)
            base += len(allsample)

    print(len(precision), len(recall))
    precision_average = np.mean(np.array(precision))
    recall_average = np.mean(np.array(recall))
    f = precision_average * recall_average * 2 / (precision_average + recall_average)
    print(precision_average, recall_average, f)


















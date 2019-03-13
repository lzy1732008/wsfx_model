from sklearn.externals import joblib
from code.model.lstm_cnn_attention import TRNNConfig,TextRNN
import tensorflow as tf
import pymysql
import jieba.posseg as pos
from code.util.wsfun import getAjjbqk,getFTfromWS,getZKDL,getRDSS
from code.process.preprocess import preprocess
import os
import re
import numpy





def readlines(path):
    alllines = []
    lines = open(path,'r',encoding= 'utf-8').read().split('\n')
    for line in lines:
        if line.strip()!='':
            alllines.append(line.strip())
    return alllines

def connectSQL():
    connection = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='83621363', db='law',
                                 charset='utf8mb4')

    # 通过cursor创建游标
    cursor = connection.cursor()
    return cursor
def getftnr(ftname,cursor):
    #remove <>/<<>>

    ftname = ftname.replace('《','')
    ftname = ftname.replace('》', '')
    ftname = ftname.replace('（', '')
    ftname = ftname.replace('）', '')
    ftname = ftname.replace('(','')
    ftname = ftname.replace(')','')
    ftname = ftname.replace('<', '')
    ftname = ftname.replace('>', '')


    start = ftname.index('第')
    end = ftname.index('条')
    ftmc = ftname[:start].strip()
    ftnum = ftname[start:end].strip()+'条'

    sql =  u"SELECT  Article_text FROM law_1_article WHERE DOC_NAME = '" + ftmc + "' AND ARTICLE_SEQ = '" + ftnum + "'"
    try:
        cursor.execute(sql)
        result = cursor.fetchone()
        return result[0]
    except:
        return ''

def feed_data(x1_batch,x2_batch, keep_prob):
    feed_dict = {
        model_relation.input_x_1: x1_batch,
        model_relation.input_x_2: x2_batch,
        model_relation.keep_prob: keep_prob
    }
    return feed_dict

def predict_ws():
    datax = []
    predicty_ws = clf.predict(datax)
    return predicty_ws

def predict_relation(datax_1,datax_2):
    feed_dict = feed_data(datax_1,datax_2,1)
    y_pred_cls = session.run(model_relation.y_pred_cls, feed_dict=feed_dict)
    return y_pred_cls

def precess(str1,stp,p,seq_length):
    words = pos.cut(str1)
    sls = []
    for word, cx in words:
        if cx == 'n' or cx == 'v' or cx == 'a':
            if word in list(stp):
                pass
            else:
                sls.append(word)
    input1 = [p.fixedvec([p.vector(ss) for ss in sls], seq_length)]
    return input1


#judge ft if has facts to support
def judgeft(ftzw,facts_data):
    datax_2 = precess(ftzw,stp,p,seq_length=10)
    print('relation')
    for datax_1 in facts_data:
        relation = predict_relation(datax_1,datax_2)
        print(relation)
        if relation[0] == 1:
            return True
    return False


def  filterft(pred_cls,facts_data):
    new_pred_ft = []
    for v,name in zip(pred_cls,ftlist):
        if v == 1:
            ftzw = getftnr(name,cursor)
            print('ftzw:',ftzw)
            if judgeft(ftzw, facts_data):
                new_pred_ft.append(name)
                continue
    return new_pred_ft

def evaluate(pred_ft,true_ft):
    print('pred_ft,trueft:',pred_ft,true_ft)
    p_t = 0
    for ft in pred_ft:
        if ft in true_ft:
            p_t += 1

    if len(pred_ft) == 0:
        return 0,0

    precision = p_t/len(pred_ft)
    recall = p_t/len(true_ft)
    return precision,recall

def setinputdata(inputtext):
    datax = []
    for key in keys:
        datax.append(inputtext.count(key))
    return datax

def predict_RF(wspath):
    ajjbqk = getAjjbqk(wspath)
    datax = setinputdata(ajjbqk)
    pred_cls =  clf.predict([datax])
    return pred_cls[0]



def ft_predict_withwsfx(wspath):
    true_ft = getFTfromWS(wspath)
    predict_cls = predict_RF(wspath)
    allfact = getZKDL(wspath) + getRDSS(wspath)
    facts = re.split('。|；',allfact)
    print('facts:',facts)
    facts_data = []
    for fact in facts:
        facts_data.append(precess(fact, stp, p, 30))

    pred_ft = filterft(predict_cls,facts_data)
    return evaluate(pred_ft,true_ft)

def ft_predict(wspath):
    true_ft = getFTfromWS(wspath)
    predict_cls = predict_RF(wspath)
    pred_ft = []
    for v,name in zip(predict_cls,ftlist):
        if v == 1:
            pred_ft.append(name)
    return evaluate(pred_ft, true_ft)



word2vec_path = '../../source/2014model_size128.model'
model_save = '../../result/model_files/事实到法条'
save_dir  = model_save + '/HNA_checkpoints/128-465-ws-30-10'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径
tensorboard_dir = model_save + '/HNA_tensorboard/128-465-ws-30-10'
config = TRNNConfig()
model_relation = TextRNN(config)
session = tf.Session()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess=session, save_path=save_path)  # 读取保存的模型


p = preprocess(word2vec_path)
p.load_models()

RF_model_path = '../../source/RF/2013model.pkl'
clf = joblib.load(RF_model_path)

stppath = '../../source/stopwords.txt'
ftpath = '../../source/RF/ftchina.txt'
keypath = '../../source/RF/ftkeys.txt'
stp = readlines(stppath)
ftlist = readlines(ftpath)
keys = readlines(keypath)
print('len(keys):',len(keys))

cursor = connectSQL()

wsdict = '/home/ftpwenshu/diskb/刑事一审案件/交通肇事罪/2014'
dir = os.listdir(wsdict)
precision,recall = [],[]
for i in range(50):
    print('index:',i)
    ws = dir[i]
    wspath = wsdict + '/' + ws
    pre_i,recall_i = ft_predict_withwsfx(wspath)
    precision.append(pre_i)
    recall.append(recall_i)

p = numpy.mean(precision,axis=0)
r = numpy.mean(recall,axis=0)
f = p*r*2/(p+r)
print(p,r,f)





































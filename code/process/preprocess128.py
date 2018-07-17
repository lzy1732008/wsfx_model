from gensim.models import word2vec
import gensim
import numpy
import jieba.analyse as ana
import random


class preprocess():
    def __init__(self,modelpath):
        self.model_path = modelpath

    def load_models(self):
        self.model = gensim.models.Word2Vec.load(self.model_path)

    def setinputdatapath(self,datapath):
        self.inputdatapath = datapath

    def setbjpath(self,datapath):#补集
        self.bjpath = datapath

    def setbj(self):
        f = open(self.bjpath,'r',encoding='utf-8')
        s = f.read().split('\n')
        f.close()
        self.bj = s

    def fixedvec(self,data,seq_length):
        if (len(data) >= seq_length):
            data = data[:seq_length]
        else:
            miss = [[0] * 128 for _ in range(seq_length - len(data))]
            data.extend(miss)
        return numpy.array(data)

    def vector(self,v):
        try:
            return self.model[v]
        except:
            return [0]*128


    def setinputdata(self,seq1_length,seq2_length,flag):
        data_1 = []
        data_2 = []
        output = []
        num1,num2 = 0, 0

        if flag == 0:#生成训练数据
            datapath = self.inputdatapath
        else:
            datapath = self.testdatapath

        f = open(datapath, 'r', encoding='utf-8')
        lines = f.read().split('\n')
        for line in lines:
            if line.strip() != '':

                ftls = (line.split('|'))[0].split(' ')
                ssls = (line.split('|'))[1].split(' ')
                label = (line.split('|'))[2]
                if label.strip()[0] == '0':
                    data_1.append(self.fixedvec([self.vector(ss) for ss in ssls], seq1_length))
                    data_2.append(self.fixedvec([self.vector(ft) for ft in ftls], seq2_length))
                    output.append([1, 0])
                    num1 += 1
                else:
                    data_1.append(self.fixedvec([self.vector(ss) for ss in ssls], seq1_length))
                    data_2.append(self.fixedvec([self.vector(ft) for ft in ftls], seq2_length))
                    output.append([0, 1])
                    num2 += 1
        print(num1)
        print(num2)
        print(len(data_1))
        return numpy.array(data_1),numpy.array(data_2),numpy.array(output)

    def settestdatapath(self,datapath):
        self.testdatapath = datapath




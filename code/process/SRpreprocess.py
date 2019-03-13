from gensim.models import word2vec
import gensim
import numpy
from code.util import matrixop
import jieba.analyse as ana
import random


class preprocess():
    def __init__(self,modelpath):
        self.model_path = modelpath

    def load_models(self):
        self.model = gensim.models.Word2Vec.load(self.model_path)

    def setinputdatapath(self,datapath):
        self.inputdatapath = datapath

    def settestdatapath(self,datapath):
        self.testdatapath = datapath

    def setvalidatedatapath(self, datapath):
        self.validatedatapath = datapath

    def setDim(self, dim):
        self.dim = dim

    def setInputLength(self, seq1_length):
        self.seq1_length = seq1_length

    def fixedvec(self,data,seq_length):
        if (len(data) >= seq_length):
            data = data[:seq_length]
        else:
            miss = [[0] * self.dim for _ in range(seq_length - len(data))]
            data.extend(miss)
        return numpy.array(data)

    def vector(self,v):
        try:
            return self.model[v]
        except:
            return [0]*self.dim


    def setinputdata(self,flag):
        data = []
        output = []

        if flag == 0:#生成训练数据
            datapath = self.inputdatapath
        elif flag == 1:#
            datapath = self.validatedatapath
        else:
            datapath = self.testdatapath

        f = open(datapath, 'r', encoding='utf-8')
        lines = f.read().split('\n')
        for line in lines:
            if line.strip() != '':
                ssls = (line.split('|'))[0].split(' ')
                ssls = list(filter(lambda x: x.strip() != '', ssls))
                label = (line.split('|'))[2].split(' ')
                label = list(filter(lambda x:x.strip()!='',label))
                data.append(self.fixedvec([self.vector(ss) for ss in ssls], self.seq_length))
                output.append([int(x.strip() for x in label)])

        print(len(data))
        return numpy.array(data),numpy.array(output)

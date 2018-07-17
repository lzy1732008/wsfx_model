# coding: utf-8

from __future__ import print_function

import os
import tensorflow as tf
import tensorflow.contrib.keras as kr
from code.process.preprocess128 import preprocess


# from cnn_word2vec.selfattention_CNN import TCNNConfig, TextCNN
# from cnn_word2vec.cnn_model_word2vec import TCNNConfig,TextCNN
# from cnn_word2vec.lstm_cnn_attention import TextRNN,TRNNConfig
from code.model.cnn_model_word2vec import TextCNN,TCNNConfig

try:
    bool(type(unicode))
except NameError:
    unicode = str


class CnnModel:
    def __init__(self,save_path):
        self.config = TCNNConfig()
        self.model = TextCNN(self.config)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型

    def predict(self, input1, input2):#input1是事实或者结论，input2是法条，
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运
        feed_dict = {
            self.model.input_x_1: kr.preprocessing.sequence.pad_sequences(input1, self.config.seq_length_1),
            self.model.input_x_2: kr.preprocessing.sequence.pad_sequences(input2, self.config.seq_length_2),
            self.model.keep_prob: 1.0
        }

        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        return y_pred_cls


#
# if __name__ == '__main__':
#
#     word2vecpath = '2014model_size64.model'
#     p = preprocess(word2vecpath)
#     p.load_models()
#
#     save_dir = '../bt/checkpoints/textselfattention_cnn_0711_64bit'
#     save_path = os.path.join(save_dir, 'best_validation')
#     cnn_model = CnnModel(save_path)
#     input1 = ['刑法 条 拘役 缓刑 期限 不能 少于 缓刑 期限 不能 少于 缓刑 期限 判决 确定 计算']
#     input2 = ['如 不服 判决 可 接到 判决书 起 内 提出 上诉']
#     data_1 = []
#     data_2 = []
#     for i in range(len(input1)):
#         data_1.append(p.fixedvec([p.vector(ss) for ss in input1], 30))
#         data_2.append(p.fixedvec([p.vector(ft) for ft in input2], 50))
#     data_1.extend(data_1)
#     data_2.extend(data_2)
#     print(cnn_model.predict(data_1,data_2))
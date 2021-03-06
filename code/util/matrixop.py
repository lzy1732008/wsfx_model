import numpy
import tensorflow as tf

def softmax(m):#0表示对行求softmax
    # print('m',m)
    m = numpy.array(m,dtype=numpy.float128)
    d = numpy.sum(numpy.exp(m),0)
    # print('numpy.exp(m)',numpy.exp((m)))
    # print('d',d)
    if d == 0:
        return numpy.zeros(shape=m.shape)
    else:
        return numpy.exp(m)/numpy.sum(numpy.exp(m),0)

def softmaxMatrix(m):
    matrix = []
    for n in m:
        matrix.append(softmax(n).tolist())
    return numpy.array(matrix)

def sumMatrix(m,axis = 0):#默认是对列求和,0是对列求和,1是对行求和
    return numpy.sum(m,axis)

def reduce_mean():
    m1 = tf.Variable(tf.random_normal([2,3],stddev=0.35),name='m1')
    m2 = tf.Variable(tf.random_normal([2,3], stddev=0.25),name='m2')
    sum = tf.add(m1,m2)/2
    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)
        print(sess.run(m1))
        print(sess.run(m2))
        print(sess.run(sum))

# reduce_mean()

#
# m = [[[0,1,2],[3,4,5],[6,7,8]],[[9,10,11],[12,13,14],[15,16,17]],[[17,18,19],[20,21,22],[22,23,24]]]
# # m = [0,1,2]
# n = sumMatrix(m,0)
# print(n)
#
#
# m1=numpy.array([[0,1,2],[4,5,6]])
# print(numpy.sum(m1,1))
# # m2=numpy.array([[7,8,9],[10,11,12]])
# # print(numpy.concatenate((m1,m2),axis=0))
# for i in range(2,4):
#     m1[i] = m1[i]*i
# print(m1)



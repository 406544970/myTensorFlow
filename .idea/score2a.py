import random

import tensorflow as tf

random.seed()

# 计算次数
times: int = 500

x = tf.placeholder(dtype=tf.float32)
yTrain = tf.placeholder(dtype=tf.float32)

# 生成全0矩阵
w = tf.Variable(tf.zeros([3]), dtype=tf.float32)

# 生成全1矩阵，其结果与生成全0矩阵基本一致
# w = tf.Variable(tf.ones([3]), dtype=tf.float32)

# 把一个向量规范后得到一个新的向量，这个新的向量中的所有数值相加起来保证为1，该函数的特性经常补用来在神经网络中处理分类的问题。
wn = tf.nn.softmax(w)

n1 = x * wn
# 计算输入tensor元素的和，或者安照reduction_indices指定的轴进行求和
n2 = tf.reduce_sum(n1)
y = tf.sigmoid(n2)

loss = tf.abs(y - yTrain)

optimizer = tf.train.RMSPropOptimizer(0.1)

train = optimizer.minimize(loss)

sess = tf.Session()

init = tf.global_variables_initializer()

sess.run(init)

for i in range(times):
    xData = [int(random.random() * 8 + 93), int(random.random() * 8 + 93), int(random.random() * 8 + 93)]
    xAll = xData[0] * 0.6 + xData[1] * 0.3 + xData[2] * 0.1
    if xAll >= 95:
        yTrainData = 1
    else:
        yTrainData = 0
    result = sess.run([train, x, yTrain, wn, n2, y, loss], feed_dict={x: xData, yTrain: yTrainData})
    print(result)

    xData = [int(random.random() * 41 + 60), int(random.random() * 41 + 60), int(random.random() * 41 + 60)]
    xAll = xData[0] * 0.6 + xData[1] * 0.3 + xData[2] * 0.1
    if xAll >= 95:
        yTrainData = 1
    else:
        yTrainData = 0
    result = sess.run([train, x, yTrain, wn, n2, y, loss], feed_dict={x: xData, yTrain: yTrainData})
    print(result)

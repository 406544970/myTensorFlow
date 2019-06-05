import random

import numpy as np
import tensorflow as tf

random.seed()

rowCount: int = 5  # 数据行数
exeCount: int = 200  # 循环次数
goodCount: int = 0  # 合格数量

xData = np.full(shape=(rowCount, 3), fill_value=0, dtype=np.float)
yTrainData = np.full(shape=rowCount, fill_value=0, dtype=np.float)

# 生成随机训练数据的循环
for i in range(rowCount):
    xData[i][0] = int(random.random() * 11 + 90)
    xData[i][1] = int(random.random() * 11 + 90)
    xData[i][2] = int(random.random() * 11 + 90)

    xAll = xData[i][0] * 0.6 + xData[i][1] * 0.1 + xData[i][2] * 0.3

    if xAll >= 95:
        yTrainData[i] = 1
        goodCount += 1
    else:
        yTrainData[i] = 0

print("xData=%s" % xData)
print("yTrainData=%s" % yTrainData)
print("goodCount=%d" % goodCount)

x = tf.placeholder(dtype=tf.float32)
yTrain = tf.placeholder(dtype=tf.float32)

w = tf.Variable(tf.zeros(3), dtype=tf.float32)
b = tf.Variable(80, dtype=tf.float32)

wn = tf.nn.softmax(w) # 生成和为1的数组
n1 = wn * x
n2 = tf.reduce_sum(n1) - b
y = tf.nn.sigmoid(n2)  # 将n2变成0到1之间的数字
loss = tf.abs(y - yTrain)

optimizer = tf.train.RMSPropOptimizer(0.1)

train = optimizer.minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(exeCount):
    for j in range(rowCount):
        result = sess.run([train, x, yTrain, wn, b, n2, y, loss]
                          , feed_dict={x: xData[j], yTrain: yTrainData[j]})
        print(result)

import random
import tensorflow as tf

random.seed()

x = tf.placeholder(dtype=tf.float32)
yTrain = tf.placeholder(dtype=tf.float32)

# 函数用于从服从指定正太分布的数值中取出指定个数的值。是概率学中的知识。
# tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
# shape: 输出张量的形状，必选
# mean: 正态分布的均值，默认为0
# stddev: 正态分布的标准差，默认为1.0
# dtype: 输出的类型，默认为tf.float32
# seed: 随机数种子，是一个整数，当设置之后，每次生成的随机数都一样
# name: 操作的名称
w = tf.Variable(tf.random_normal([4], mean=0.5, stddev=0.1), dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)

n1 = w * x + b
y = tf.nn.sigmoid(tf.reduce_sum(n1))

loss = tf.abs(y - yTrain)
optimizer = tf.train.RMSPropOptimizer(0.01)
train = optimizer.minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

lossSum: float = 0.0
for i in range(10):
    xDataRandom = [int(random.random() * 10)
        , int(random.random() * 10)
        , int(random.random() * 10)
        , int(random.random() * 10)]
    if xDataRandom[2] % 2 == 0:
        yTrainDataRandom = 0
    else:
        yTrainDataRandom = 1

    result = sess.run([train, x, yTrain, w,y, loss]
                      , feed_dict={x: xDataRandom, yTrain: yTrainDataRandom})
    print(result)

    lossSum += float(result[len(result) - 1])
    print("i:%d,loss:%10.10f,avgLoss:%10.10f" % (i
                                                 , float(result[len(result) - 1])
                                                 , lossSum / (i + 1)))

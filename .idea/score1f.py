import tensorflow as tf

# 计算次数
times: int = 30000

x = tf.placeholder(shape=[3], dtype=tf.float32)
yTrain = tf.placeholder(shape=[], dtype=tf.float32)

# 生成全0矩阵
w = tf.Variable(tf.zeros([3]), dtype=tf.float32)

# 生成全1矩阵，其结果与生成全0矩阵基本一致
# w = tf.Variable(tf.ones([3]), dtype=tf.float32)

# 把一个向量规范后得到一个新的向量，这个新的向量中的所有数值相加起来保证为1，该函数的特性经常补用来在神经网络中处理分类的问题。
wn = tf.nn.softmax(w)

n = x * wn

# 计算输入tensor元素的和，或者安照reduction_indices指定的轴进行求和
y = tf.reduce_sum(n)

loss = tf.abs(y - yTrain)

optimizer = tf.train.RMSPropOptimizer(0.0001)

train = optimizer.minimize(loss)

sess = tf.Session()

init = tf.global_variables_initializer()

sess.run(init)

for i in range(times):
    result = sess.run([train, x, w, wn, y, yTrain, loss], feed_dict={x: [90, 80, 70], yTrain: 85})
    print(result)

    result = sess.run([train, x, w, wn, y, yTrain, loss], feed_dict={x: [98, 95, 87], yTrain: 96})
    print(result)

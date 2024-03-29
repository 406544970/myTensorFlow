import tensorflow as tf

# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息  
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error   
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error

x1 = tf.placeholder(dtype=tf.float32)
x2 = tf.placeholder(dtype=tf.float32)
x3 = tf.placeholder(dtype=tf.float32)
yTrain = tf.placeholder(dtype=tf.float32)

w1 = tf.Variable(0.1, dtype=tf.float32)
w2 = tf.Variable(0.2, dtype=tf.float32)
w3 = tf.Variable(0.3, dtype=tf.float32)

n1 = x1 * w1
n2 = x2 * w2
n3 = x3 * w3

y = n1 + n2 + n3

loss = tf.abs(y - yTrain)

optimizer = tf.train.RMSPropOptimizer(0.001)

train = optimizer.minimize(loss)

sess = tf.Session()

init = tf.global_variables_initializer()

sess.run(init)

for i in range(10):
    result = sess.run([train, x1, x2, x3, w1, w2, w3, y, yTrain, loss]
                      , feed_dict={x1: 92, x2: 98, x3: 90, yTrain: 94})
    print(result)

    result = sess.run([train, x1, x2, x3, w1, w2, w3, y, yTrain, loss]
                      , feed_dict={x1: 92, x2: 99, x3: 98, yTrain: 96})
    print(result)

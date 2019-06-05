import tensorflow as tf

xData = tf.constant([
    [[0],[0],[1],[1],[0],[0]],
    [[0],[1],[0],[0],[1],[0]],
    [[0],[0],[0],[0],[1],[0]],
    [[0],[0],[0],[1],[0],[0]],
    [[0],[0],[1],[0],[0],[0]],
    [[0],[1],[1],[1],[1],[0]]
]
                    ,dtype=tf.float32)
filterT = tf.constant([[1,2],[3,4]],dtype=tf.float32)

y = tf.nn.conv2d(tf.reshape(xData,[1,6,6,1]),tf.reshape(filterT,[2,2,1,1])
                 ,strides=[1,1,1,1],padding="VALID")
sess = tf.Session()
result = sess.run(y)
print(result)
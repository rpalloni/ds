'''
a tensor is a vector/matrix of n dimension that represents all type of data

    | 1 |       | 2  1 |        | | 1 2 | | 5 6 | |
    | 2 |       | 4  3 |        | | 3 4 | | 8 4 | |
    | 3 |       | 6  4 |        | | 3 1 | | 2 2 | |
    vector       matrix               tensor
'''
import tensorflow as tf
print(tf.__version__)

# constant tensor
C = tf.constant([[4, 3], [6, 1]])

# variable tensor
V = tf.Variable([[3, 1], [5, 2]])


# concat
concat_row = tf.concat(values=[C, V], axis=0)
concat_col = tf.concat(values=[C, V], axis=1)

tf.print(concat_col)

# dummy tensor
Z = tf.zeros(shape=[3, 4], dtype=tf.int32)
O = tf.ones(shape=[3, 4], dtype=tf.int32)
Z

# random
R = tf.random.uniform(shape=[3, 4], dtype=tf.float32)
R

# transpose
T = tf.transpose(R)
T

# matrix multiplication
A = tf.constant([[4, 3], [6, 1]]) # 2x2
v = tf.constant([[2], [4]]) # 2x1
Av = tf.matmul(A, v) # 2x1
Av

# math utils: tf.math
x = tf.constant(10.0, name='x', dtype=tf.float32)
y = tf.constant(20.0, name='y', dtype=tf.float32)
z = tf.Variable(tf.add(x, y))

tf.print(z)

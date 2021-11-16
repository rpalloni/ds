'''
https://www.tensorflow.org/guide/tensor
A tensor is a generalization of vectors and matrices to potentially higher dimensions.
TF represents tensors as n-dimensional arrays of base datetypes.
Each tensor has three attributes: datatype (float32, int32, string, etc), shape (axes size) and rank (number of dim)

                | 1 |       | 2  1 |        | | 1 2 | | 5 6 | |
    | 1 |       | 2 |       | 4  3 |        | | 3 4 | | 8 4 | |
                | 3 |       | 6  4 |        | | 3 1 | | 2 2 | |
    scalar      vector       matrix               tensor
    rank 0      rank 1       rank 2               rank n
'''
import tensorflow as tf
print(tf.__version__)

string = tf.Variable('hello world', tf.string)
print(tf.rank(string)) # rank 0 tensor => scalar
string.shape # empty
integer = tf.Variable(123, tf.int16)
float = tf.Variable(3.45, tf.float64)

s = tf.Variable(['hello world'], tf.string)
print(tf.rank(s)) # rank 1 tensor => vector
s.shape

# constant tensor (immutable)
C = tf.constant([[4, 3], [6, 1]])
C

# variable tensor (mutable)
V = tf.Variable([[3, 1], [5, 2]])
print(tf.rank(V)) # rank 2 tensor => matrix

T = tf.Variable([[[3, 1, 4], [5, 2, 7]], [[4, 8, 2], [6, 3, 0]]])
print(tf.rank(T))
T.shape
T

# concat
concat_row = tf.concat(values=[C, V], axis=0)
concat_col = tf.concat(values=[C, V], axis=1)

tf.print(concat_col)

# reshape
R = tf.reshape(T, shape=[3, 2, 2]) # outer to inner sizes of lists
R

W = tf.reshape(T, shape=[4, -1]) # -1: infer the shape given a list size
W

# dummy tensor
Z = tf.zeros(shape=[3, 4], dtype=tf.int32)
Z
O = tf.ones(shape=[5, 5, 5], dtype=tf.int32)
O

H = tf.reshape(O, shape=[125]) # flatten values
H

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

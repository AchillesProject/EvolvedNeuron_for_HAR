import tensorflow as tf
import time

tf.random.set_seed(42)
A = tf.random.normal([10000, 10000])
B = tf.random.normal([10000, 10000])

def check():
    start_time = time.time()
    tf.math.reduce_sum(tf.linalg.matmul(A,B))
    print("It took {} seconds".format(time.time() - start_time))

check()

import tensorflow as tf
import numpy as np


def mullAdd(num_add: float):
    return lambda x, a: a * x + num_add


if __name__ == "__main__":
    x = lambda a: a + 1
    print(x(5))

    x1 = mullAdd(10)
    print(x1(5, 5))

    x2 = tf.Variable(np.arange(12).reshape(3, -1), dtype=np.float32, name="x2")

    max_loop = tf.shape(x2)[0]

    t1 = tf.constant(1)
    t2 = tf.constant(2)


    def f1(): return t1 + t2


    def f2(): return t1 - t2


    print(tf.cond(tf.less(t1, t2), f2, f1))

    x = tf.Variable(10)
    y = tf.Variable(5)

    print(tf.cond(x < y, lambda: tf.add(x, 10), lambda: tf.maximum(x, 20)))


    def body(t1, t2, i, imax):
        t1 -= 1
        print(i)
        i = i + 1
        return [t1, t2, i, imax]


    print(tf.while_loop(lambda t1, t2, i, imax: i < imax, body, [7, 5, 1, 10]))

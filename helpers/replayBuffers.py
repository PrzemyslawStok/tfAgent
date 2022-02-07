import random
from collections import deque

import tensorflow as tf
from tf_agents.replay_buffers import tf_uniform_replay_buffer

import numpy as np

if __name__ == "__main__":

    data_spec = tf.TensorSpec([3], tf.float32, "action")
    batch_size = 2
    max_length = 1000

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=data_spec,
        batch_size=batch_size,
        max_length=max_length
    )

    print(replay_buffer.data_spec)

    value = tf.ones([3])
    batched1 = tf.expand_dims(value, axis=0)
    batched = tf.nest.map_structure(lambda t: tf.stack([t] * batch_size), value)

    batched2 = tf.stack([value])

    # Version with ragged tensor

    #replay_buffer.add_batch(batched)

    for i in range(5):
        value = tf.ones([3]) * i
        #batch = tf.nest.map_structure(lambda x: tf.stack[x] * batch_size, value)
        replay_buffer.add_batch(tf.stack([value]*batch_size))



    replay_deque = deque(maxlen=100)

    for i in range(100):
        replay_deque.append(value)

    replay_buffer.add_batch(tf.stack(random.sample(replay_deque, batch_size)))


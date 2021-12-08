import tensorflow as tf

import PIL.Image
import pyvirtualdisplay

import numpy as np

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer, tf_uniform_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

import time

def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]


# See also the metrics module for standard implementations of different metrics.
# https://github.com/tensorflow/agents/tree/master/tf_agents/metrics


if __name__ == '__main__':
    print('PyCharm')

    # Set up a virtual display for rendering OpenAI gym environments.
    display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

    print(tf.version.VERSION)

    num_iterations = 20000

    initial_collect_steps = 100
    collect_steps_per_iteration = 1
    replay_buffer_max_length = 100000
    batch_size = 64
    learning_rate = 1e-3
    log_interval = 200

    num_eval_episodes = 10
    eval_interval = 1000

    env_name = 'CartPole-v0'
    env = suite_gym.load(env_name)

    env.reset()
    PIL.Image.fromarray(env.render())

    print('Observation Spec:')
    print(env.time_step_spec().observation)

    print('Reward Spec:')
    print(env.time_step_spec().reward)

    print('Action Spec:')
    print(env.action_spec())

    time_step = env.reset()
    print('Time step:')
    print(time_step)

    action = np.array(1, dtype=np.int32)

    for i in range(0):
        next_time_step = env.step(action)
        #print('Next time step:')
        print(next_time_step.reward)
        time.sleep(0.1)
        PIL.Image.fromarray(env.render())

    #train_py_env = suite_gym.load(env_name)
    #eval_py_env = suite_gym.load(env_name)

    train_env = tf_py_environment.TFPyEnvironment(suite_gym.load('CartPole-v0'))
    eval_env = tf_py_environment.TFPyEnvironment(suite_gym.load('CartPole-v0'))

    fc_layer_params = (100, 50)
    action_tensor_spec = tensor_spec.from_spec(env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1


    # Define a helper function to create Dense layers configured with the right
    # activation and kernel initializer.
    def dense_layer(num_units):
        return tf.keras.layers.Dense(
            num_units,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, mode='fan_in', distribution='truncated_normal'))


    # QNetwork consists of a sequence of Dense layers followed by a dense layer
    # with `num_actions` units to generate one q_value per available action as
    # its output.
    dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.03, maxval=0.03),
        bias_initializer=tf.keras.initializers.Constant(-0.2))
    q_net = sequential.Sequential(dense_layers + [q_values_layer])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    train_step_counter = tf.Variable(0)

    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)

    agent.initialize()

    eval_policy = agent.policy
    collect_policy = agent.collect_policy

    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                    train_env.action_spec())

    example_environment = tf_py_environment.TFPyEnvironment(
        suite_gym.load('CartPole-v0'))

    time_step = example_environment.reset()

    print("sum: ",compute_avg_return(eval_env, random_policy, num_eval_episodes))

    table_name = 'uniform_table'
    replay_buffer_signature = tensor_spec.from_spec(
        agent.collect_data_spec)
    replay_buffer_signature = tensor_spec.add_outer_dim(
        replay_buffer_signature)













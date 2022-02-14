import os
import tempfile

from PIL import Image
import gym
import imageio
import tensorflow as tf
import tf_agents
from matplotlib import pyplot as plot
from tensorflow.keras.utils import Progbar
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver, dynamic_episode_driver, py_driver
from tf_agents.environments import ParallelPyEnvironment, suite_pybullet
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy, policy_saver, py_tf_eager_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.train.utils import strategy_utils, spec_utils
from tf_agents.utils import common

import timeit

tempdir = tempfile.gettempdir()

env_name = "MinitaurBulletEnv-v0" # @param {type:"string"}

# Use "num_iterations = 1e6" for better results (2 hrs)
# 1e5 is just so this doesn't take too long (1 hr)
num_iterations = 100000 # @param {type:"integer"}

initial_collect_steps = 10000 # @param {type:"integer"}
collect_steps_per_iteration = 1 # @param {type:"integer"}
replay_buffer_capacity = 10000 # @param {type:"integer"}

batch_size = 256 # @param {type:"integer"}

critic_learning_rate = 3e-4 # @param {type:"number"}
actor_learning_rate = 3e-4 # @param {type:"number"}
alpha_learning_rate = 3e-4 # @param {type:"number"}
target_update_tau = 0.005 # @param {type:"number"}
target_update_period = 1 # @param {type:"number"}
gamma = 0.99 # @param {type:"number"}
reward_scale_factor = 1.0 # @param {type:"number"}

actor_fc_layer_params = (256, 256)
critic_joint_fc_layer_params = (256, 256)

log_interval = 5000 # @param {type:"integer"}

num_eval_episodes = 20 # @param {type:"integer"}
eval_interval = 10000 # @param {type:"integer"}

policy_save_interval = 5000 # @param {type:"integer"}

if __name__ == '__main__':
    env = suite_pybullet.load(env_name)
    env.reset()
    image = Image.fromarray(env.render())
    image.show()

    print('Observation Spec:')
    print(env.time_step_spec().observation)
    print('Action Spec:')
    print(env.action_spec())

    collect_env = suite_pybullet.load(env_name)
    eval_env = suite_pybullet.load(env_name)

    use_gpu = True
    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=use_gpu)

    observation_spec, action_spec, time_step_spec = (
        spec_utils.get_tensor_specs(collect_env))





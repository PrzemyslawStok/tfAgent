import os
import tempfile

from PIL import Image
import gym
import imageio
import tensorflow as tf
import tf_agents
from matplotlib import pyplot as plot
from tensorflow.keras.utils import Progbar
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.sac import tanh_normal_projection_network, sac_agent
from tf_agents.drivers import dynamic_step_driver, dynamic_episode_driver, py_driver
from tf_agents.environments import ParallelPyEnvironment, suite_pybullet
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import tf_metrics, py_metrics
from tf_agents.networks import q_network, actor_distribution_network
from tf_agents.policies import random_tf_policy, policy_saver, py_tf_eager_policy, random_py_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.train import actor, learner
from tf_agents.train.utils import strategy_utils, spec_utils, train_utils
from tf_agents.utils import common

import timeit

tempdir = tempfile.gettempdir()

env_name = "MinitaurBulletEnv-v0"  # @param {type:"string"}

# Use "num_iterations = 1e6" for better results (2 hrs)
# 1e5 is just so this doesn't take too long (1 hr)
num_iterations = 100000  # @param {type:"integer"}

initial_collect_steps = 10000  # @param {type:"integer"}
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_capacity = 10000  # @param {type:"integer"}

batch_size = 256  # @param {type:"integer"}

critic_learning_rate = 3e-4  # @param {type:"number"}
actor_learning_rate = 3e-4  # @param {type:"number"}
alpha_learning_rate = 3e-4  # @param {type:"number"}
target_update_tau = 0.005  # @param {type:"number"}
target_update_period = 1  # @param {type:"number"}
gamma = 0.99  # @param {type:"number"}
reward_scale_factor = 1.0  # @param {type:"number"}

actor_fc_layer_params = (256, 256)
critic_joint_fc_layer_params = (256, 256)

log_interval = 5000  # @param {type:"integer"}

num_eval_episodes = 20  # @param {type:"integer"}
eval_interval = 10000  # @param {type:"integer"}

policy_save_interval = 5000  # @param {type:"integer"}

if __name__ == '__main__':
    env = suite_pybullet.load(env_name)
    env.reset()
    # image = Image.fromarray(env.render())
    # image.show()

    collect_env = suite_pybullet.load(env_name)
    eval_env = suite_pybullet.load(env_name)

    use_gpu = False
    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=use_gpu)

    observation_spec, action_spec, time_step_spec = (
        spec_utils.get_tensor_specs(collect_env))

    print('Observation Spec:')
    print(observation_spec)
    print('Action Spec:')
    print(action_spec)
    print('TimeStep Spec:')
    print(time_step_spec)

    with strategy.scope():
        critic_net = critic_network.CriticNetwork(
            (observation_spec, action_spec),
            observation_fc_layer_params=None,
            action_fc_layer_params=None,
            joint_fc_layer_params=critic_joint_fc_layer_params,
            kernel_initializer='glorot_uniform',
            last_kernel_initializer='glorot_uniform')

        actor_net = actor_distribution_network.ActorDistributionNetwork(
            observation_spec,
            action_spec,
            fc_layer_params=actor_fc_layer_params,
            continuous_projection_net=(
                tanh_normal_projection_network.TanhNormalProjectionNetwork))

        train_step = train_utils.create_train_step()

        tf_agent = sac_agent.SacAgent(
            time_step_spec,
            action_spec,
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=tf.keras.optimizers.Adam(
                learning_rate=actor_learning_rate),
            critic_optimizer=tf.keras.optimizers.Adam(
                learning_rate=critic_learning_rate),
            alpha_optimizer=tf.keras.optimizers.Adam(
                learning_rate=alpha_learning_rate),
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            td_errors_loss_fn=tf.math.squared_difference,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            train_step_counter=train_step)

        tf_agent.initialize()

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=tf_agent.collect_data_spec,
            batch_size=1,
            max_length=replay_buffer_capacity)

        replay_observer = [replay_buffer.add_batch]

        dataset = replay_buffer.as_dataset(
            sample_batch_size=batch_size,
            num_steps=2).prefetch(50)

        experience_dataset_fn = lambda: dataset

        tf_eval_policy = tf_agent.policy
        eval_policy = py_tf_eager_policy.PyTFEagerPolicy(
            tf_eval_policy, use_tf_function=True)

        tf_collect_policy = tf_agent.collect_policy
        collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
            tf_collect_policy, use_tf_function=True)

        random_policy = random_py_policy.RandomPyPolicy(
            collect_env.time_step_spec(), collect_env.action_spec())

        initial_collect_actor = actor.Actor(
            collect_env,
            random_policy,
            train_step,
            steps_per_run=initial_collect_steps,
            observers=[replay_observer])

        initial_collect_actor.run()

        env_step_metric = py_metrics.EnvironmentSteps()
        collect_actor = actor.Actor(
            collect_env,
            collect_policy,
            train_step,
            steps_per_run=1,
            metrics=actor.collect_metrics(10),
            summary_dir=os.path.join(tempdir, learner.TRAIN_DIR),
            observers=[replay_observer, env_step_metric])




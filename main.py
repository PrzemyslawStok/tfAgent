import base64
import IPython
import imageio

import tensorflow as tf

import PIL.Image
import pyvirtualdisplay

import numpy as np

from ConstructQnetwork import construct_qnet

from tf_agents.agents import tf_agent

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver, dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential, q_network
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer, tf_uniform_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

from matplotlib import pyplot as plot

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


def create_policy_eval_video(eval_py_env, policy, filename, num_episodes=5, fps=30):
    filename = filename + ".mp4"
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    with imageio.get_writer(filename, fps=fps) as video:
        for _ in range(num_episodes):
            time_step = eval_env.reset()
            video.append_data(eval_py_env.render())
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = eval_env.step(action_step.action)
                video.append_data(eval_py_env.render())
    return filename


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

    mountains = "MountainCar-v0"
    cartpole = "CartPole-v1"
    env_name = cartpole
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
        # print('Next time step:')
        print(next_time_step.reward)
        time.sleep(0.1)
        PIL.Image.fromarray(env.render())

    # train_py_env = suite_gym.load(env_name)
    # eval_py_env = suite_gym.load(env_name)

    train_env = tf_py_environment.TFPyEnvironment(suite_gym.load(env_name))
    eval_env = tf_py_environment.TFPyEnvironment(suite_gym.load(env_name))

    fc_layer_params = (100, 100, 50)
    action_tensor_spec = tensor_spec.from_spec(env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

    q_net = construct_qnet(num_actions, fc_layer_params)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    train_step_counter = tf.Variable(0)

    #q_net = q_network.QNetwork(
    #    train_env.observation_spec(),
    #    train_env.action_spec(),
    #    fc_layer_params=fc_layer_params)

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
        suite_gym.load(env_name))

    time_step = example_environment.reset()

    print("sum: ", compute_avg_return(eval_env, random_policy, num_eval_episodes))

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length)

    replay_observer = [replay_buffer.add_batch]

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=8,
        sample_batch_size=batch_size,
        num_steps=2).prefetch(3)

    iterator = iter(dataset)

    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
    ]

    driver = dynamic_step_driver.DynamicStepDriver(
        train_env,
        collect_policy,
        observers=replay_observer + train_metrics,
        num_steps=1)

    episode_len = []

    final_time_step, policy_state = driver.run()

    print(final_time_step, policy_state)

    eval_py_env = suite_gym.load(env_name)
    create_policy_eval_video(eval_py_env, agent.policy, env_name + "-untrainded-agent")

    for i in range(num_iterations):
        final_time_step, _ = driver.run(final_time_step, policy_state)

        experience, _ = next(iterator)
        train_loss = agent.train(experience=experience)
        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss.loss))
            episode_len.append(train_metrics[3].result().numpy())
            print('Average episode length: {}'.format(train_metrics[3].result().numpy()))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))

    plot.plot(episode_len)
    plot.show()

    create_policy_eval_video(eval_py_env, agent.policy, env_name + "-trained-agent")

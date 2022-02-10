import tensorflow as tf
from tf_agents.drivers import dynamic_step_driver, dynamic_episode_driver
from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.metrics import tf_metrics
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory

from CustomEnv.GraphEnv import sampleEnv


def create_buffer(env: tf_py_environment, batch_size=1,
                  max_length=100) -> tf_uniform_replay_buffer.TFUniformReplayBuffer:

    trajectory_spec = trajectory.Trajectory(
        step_type=env.time_step_spec().step_type,
        observation=env.time_step_spec().observation,
        action=env.action_spec(),
        policy_info=(),
        next_step_type=env.time_step_spec().step_type,
        reward=env.time_step_spec().reward,
        discount=env.time_step_spec().discount)

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(trajectory_spec, batch_size=batch_size,
                                                                   max_length=max_length)

    return replay_buffer


def stepDriver(env: tf_py_environment, replay_buffer: tf_uniform_replay_buffer.TFUniformReplayBuffer):
    policy = random_tf_policy.RandomTFPolicy(action_spec=env.action_spec(), time_step_spec=env.time_step_spec())

    steps = tf_metrics.EnvironmentSteps()
    episodes = tf_metrics.NumberOfEpisodes()
    reward = tf_metrics.AverageReturnMetric()
    driver_observers = [replay_buffer.add_batch, steps, episodes, reward]
    driver = dynamic_step_driver.DynamicStepDriver(env, policy=policy, observers=driver_observers, num_steps=100_000)

    initial_time_step = env.reset()
    driver.run(initial_time_step)

    print(f"reward: {reward.result().numpy()}")
    print(f"steps: {steps.result().numpy()}")
    print(f"episodes: {episodes.result().numpy()}")


if __name__ == "__main__":
    #env = suite_gym.load('CartPole-v0')
    #tf_env = tf_py_environment.TFPyEnvironment(env)

    tf_env: tf_py_environment = tf_py_environment.TFPyEnvironment(sampleEnv())

    replay_buffer = create_buffer(tf_env)
    stepDriver(tf_env, replay_buffer)

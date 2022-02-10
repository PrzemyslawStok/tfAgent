import tensorflow as tf
from tf_agents.drivers import dynamic_step_driver, dynamic_episode_driver
from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.metrics import tf_metrics
from tf_agents.policies import random_tf_policy


def stepDriver(env: tf_py_environment):
    policy = random_tf_policy.RandomTFPolicy(action_spec=env.action_spec(), time_step_spec=env.time_step_spec())

    steps = tf_metrics.EnvironmentSteps()
    episodes = tf_metrics.NumberOfEpisodes()
    reward = tf_metrics.AverageReturnMetric()
    driver_observers = [steps, episodes, reward]
    driver = dynamic_step_driver.DynamicStepDriver(env, policy=policy, observers=driver_observers, num_steps=100)

    initial_time_step = env.reset()
    driver.run(initial_time_step)

    print(f"reward: {reward.result().numpy()}")
    print(f"steps: {steps.result().numpy()}")

if __name__ == "__main__":
    env = suite_gym.load('CartPole-v0')
    tf_env = tf_py_environment.TFPyEnvironment(env)

    stepDriver(tf_env)

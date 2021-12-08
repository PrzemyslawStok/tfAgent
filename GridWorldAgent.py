import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer

from tf_agents.utils import common

from GridWorldEnv import GridWorldEnv
from tf_agents.environments import py_environment, utils, wrappers, tf_py_environment

from matplotlib import pyplot as plot

def f0(environment, policy, num_episodes=10):
    pass

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

if __name__ == '__main__':
    env = GridWorldEnv()
    utils.validate_py_environment(env, episodes=5)

    train_py_env = wrappers.TimeLimit(GridWorldEnv(), duration=100)
    eval_py_env = wrappers.TimeLimit(GridWorldEnv(), duration=100)

    train_env = tf_py_environment.TFPyEnvironment(wrappers.TimeLimit(GridWorldEnv(), duration=100))
    eval_env = tf_py_environment.TFPyEnvironment(wrappers.TimeLimit(GridWorldEnv(), duration=100))

    fc_layer_params = (100,)

    q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params)

    learning_rate = 1e-3
    replay_buffer_capacity = 10000
    batch_size = 128
    num_iterations = 1000
    log_interval = 200
    eval_interval = 1000
    num_eval_episodes = 2

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    train_step_counter = tf.compat.v2.Variable(0)

    tf_agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)

    tf_agent.initialize()

    eval_policy = tf_agent.policy
    collect_policy = tf_agent.collect_policy

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_capacity)

    replay_observer = [replay_buffer.add_batch]

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
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

    for i in range(num_iterations):
        final_time_step, _ = driver.run(final_time_step, policy_state)

        experience, _ = next(iterator)
        train_loss = tf_agent.train(experience=experience)
        step = tf_agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss.loss))
            episode_len.append(train_metrics[3].result().numpy())
            print('Average episode length: {}'.format(train_metrics[3].result().numpy()))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))

    plot.plot(episode_len)
    plot.show()
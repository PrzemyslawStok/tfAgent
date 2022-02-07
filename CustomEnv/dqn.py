import tensorflow as tf
from keras.utils.generic_utils import Progbar
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import tf_metrics

from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

from GraphEnv import sampleEnv

replay_buffer_max_length = 100000
batch_size = 32
learning_rate = 1e-3

num_iterations = 100
log_interval  = 10

random_policy = False

def create_qnet(fc_layer_params: tuple, train_env: tf_py_environment) -> q_network:
    return q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params)


if __name__ == "__main__":
    # można utworzyć nowe środowidko funkcja sample env dodaje parametry, które były w google colab
    train_env: tf_py_environment = tf_py_environment.TFPyEnvironment(sampleEnv())

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    train_step_counter = tf.Variable(0)

    fc_layer_params = (64, 256)
    q_net = create_qnet(fc_layer_params, train_env)

    # tutaj trzeba uważać agent powinien sam utworzyć dodatkową sieć ale podobno może nie zadziałać
    target_q_net = create_qnet(fc_layer_params, train_env)

    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter,
        target_q_network=target_q_net)

    agent.initialize()

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length)

    replay_observer = [replay_buffer.add_batch]

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=2,
        sample_batch_size=batch_size,
        num_steps=2).prefetch(2)

    dataset_iterator = iter(dataset)

    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
    ]

    if random_policy:
        policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),train_env.action_spec())
    else:
        policy = agent.collect_policy

    driver = dynamic_episode_driver.DynamicEpisodeDriver(train_env,
                                                         policy,
                                                         observers=replay_observer + train_metrics,
                                                         num_episodes=1)

    metrics_names = ['reward', 'length']
    progbar = Progbar(num_iterations, stateful_metrics=metrics_names)

    episode_len = []

    agent.train = common.function(agent.train)

    for i in range(num_iterations):
        final_time_step, _ = driver.run()

        experience, _ = next(dataset_iterator)

        if not random_policy:
            agent.train(experience)

        if i % log_interval == 0:
            episode_len.append(train_metrics[3].result().numpy())
            values = [(metrics_names[0], train_metrics[2].result().numpy()), (metrics_names[1], train_metrics[3].result().numpy())]
            progbar.update(i + 1, values)

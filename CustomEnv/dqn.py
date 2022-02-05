import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment

from tf_agents.networks import q_network
from tf_agents.utils import common

from GraphEnv import GraphEnv, sampleEnv

replay_buffer_max_length = 100000
batch_size = 64
learning_rate = 1e-3


def create_qnet(fc_layer_params: tuple, train_env: tf_py_environment) -> q_network:
    return q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params)


if __name__ == "__main__":
    # można utworzyć nowe środowidko funkcja sample env dodaje parametry, które były w google colab
    train_env: GraphEnv = sampleEnv()

    train_env_tf = tf_py_environment.TFPyEnvironment(
        train_env
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    train_step_counter = tf.Variable(0)

    fc_layer_params = (64, 256)
    q_net = create_qnet(fc_layer_params, train_env_tf)

    # tutaj trzeba uważać agent powinien sam utworzyć dodatkową sieć ale podobno może nie zadziałać
    target_q_net = create_qnet(fc_layer_params, train_env_tf)

    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter,
        target_q_network=target_q_net)

    agent.initialize()

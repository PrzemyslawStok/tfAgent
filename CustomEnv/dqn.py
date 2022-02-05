import tensorflow as tf

from tf_agents.networks import q_network
from GraphEnv import GraphEnv, sampleEnv

replay_buffer_max_length = 100000
batch_size = 64
learning_rate = 1e-3

if __name__ == "__main__":
    # można utworzyć nowe środowidko funkcja sample env dodaje parametry, które były w google colab
    train_env: GraphEnv = sampleEnv()

    fc_layer_params = (64, 256)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    train_step_counter = tf.Variable(0)

    q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params)

    # tutaj trzeba uważać agent powinien sam utworzyć dodatkową sieć ale podobno może nie zadziałać
    target_q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params)

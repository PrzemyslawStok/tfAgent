import gym
import imageio
import pyvirtualdisplay
import tensorflow as tf
import tf_agents
import os
from matplotlib import pyplot as plot
from tensorflow.keras.utils import Progbar
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.environments import ParallelPyEnvironment
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy, policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.utils import common



from ConstructQnetwork import construct_qnet
from GridWorldEnv import GridWorldEnv


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


def envInfo(env):
    print('Observation Spec:')
    print(env.time_step_spec().observation)

    print('Reward Spec:')
    print(env.time_step_spec().reward)

    print('Action Spec:')
    print(env.action_spec())

    time_step = env.reset()
    print('Time step:')
    print(time_step)

def loadAgent(agentDir: str, savedPolicy:str):
    return tf.saved_model.load(os.path.join(agentDir,savedPolicy))

def main(argv):
    num_iterations = 1000

    replay_buffer_max_length = 100000
    batch_size = 64
    learning_rate = 1e-3
    log_interval = 1000

    num_eval_episodes = 10
    parallel_calls = 1

    agentDir = "savedAgents"

    pendulum = "Pendulum-v1"
    acrobot = "Acrobot-v1"
    mountains = "MountainCarContinuous-v0"
    cartpole = "CartPole-v1"
    lunar_lander = "LunarLander-v2"
    env_name = cartpole

    env = gym.make(cartpole)
    env = suite_gym.wrap_env(env)

    envInfo(env)

    #exit()

    # env = GridWorldEnv()

    env.reset()

    train_env = tf_py_environment.TFPyEnvironment(
        ParallelPyEnvironment(
            [lambda: env] * parallel_calls
        )
    )

    eval_env = tf_py_environment.TFPyEnvironment(
        ParallelPyEnvironment(
            [lambda: env] * parallel_calls
        )
    )

    fc_layer_params = (100, 100, 50)
    action_tensor_spec = tensor_spec.from_spec(env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

    #q_net = construct_qnet(num_actions, fc_layer_params)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    train_step_counter = tf.Variable(0)

    q_net = q_network.QNetwork(
       train_env.observation_spec(),
       train_env.action_spec(),
       fc_layer_params=fc_layer_params)

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
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length)

    replay_observer = [replay_buffer.add_batch]

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=parallel_calls,
        sample_batch_size=batch_size*parallel_calls,
        num_steps=2).prefetch(parallel_calls*2)

    iterator = iter(dataset)

    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(batch_size=parallel_calls),
        tf_metrics.AverageEpisodeLengthMetric(batch_size=parallel_calls),
    ]

    driver = dynamic_step_driver.DynamicStepDriver(
        train_env,
        collect_policy,
        observers=replay_observer + train_metrics,
        num_steps=2)

    episode_len = []

    final_time_step, policy_state = driver.run()

    print(final_time_step, policy_state)

    # create_policy_eval_video(env, agent.policy, env_name + "-untrainded-agent")

    metrics_names = ['loss', 'epizode length', 'average']
    progbar = Progbar(num_iterations, stateful_metrics=metrics_names)

    compute_avg_return(train_env, agent.policy, num_eval_episodes)

    for i in range(num_iterations):
        final_time_step, _ = driver.run(final_time_step, policy_state)

        experience, _ = next(iterator)
        train_loss = agent.train(experience=experience)
        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            episode_len.append(train_metrics[3].result().numpy())
            values = [(metrics_names[0], train_loss.loss), (metrics_names[1], train_metrics[3].result().numpy()),
                      (metrics_names[2], compute_avg_return(eval_env, agent.policy, num_eval_episodes))]
            progbar.update(i + 1, values)
            plot.plot(episode_len)
            plot.show()

        if i == num_iterations - 1:
            values = [(metrics_names[0], train_loss.loss), (metrics_names[1], train_metrics[3].result().numpy()),
                      (metrics_names[2], compute_avg_return(eval_env, agent.policy, num_eval_episodes))]
            progbar.update(i + 1, values, finalize=True)

    policy_dir = os.path.join(agentDir, env_name + "-trained-agent")
    tf_policy_saver = policy_saver.PolicySaver(agent.policy)
    tf_policy_saver.save(policy_dir)

    create_policy_eval_video(env, agent.policy, env_name + "-trained-agent")

    agent = loadAgent(agentDir, "Cartpole-v1-trained-agent")


if __name__ == '__main__':
    tf_agents.system.multiprocessing.handle_main(main)

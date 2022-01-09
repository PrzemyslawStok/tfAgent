import time
import gym
import keras.models
from gym import envs
import tensorflow as tf
import os

import numpy as np

import tensorflow_probability
from tf_agents.policies import py_tf_eager_policy, random_tf_policy

from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import tensor_spec

agentDir = "savedAgents"
modelsDir = "savedModels"


def renderEnv(envName="CartPole-v0", policy=None, doneAfterEnd=False, closeEnv=False, steps=200, epizodes=1):
    env = gym.make(envName)
    env1 = env.unwrapped

    time_spec = ts.time_step_spec(suite_gym.gym_wrapper.spec_from_gym_space(env.observation_space))

    if policy == None:
        policy = random_tf_policy.RandomTFPolicy(time_spec,
                                                 suite_gym.gym_wrapper.spec_from_gym_space(env.action_space))

    for episode in range(epizodes):
        observation = env.reset()
        reward = 0.0

        for i in range(steps):
            env.render()

            time_step = ts.transition(tf.expand_dims(tf.convert_to_tensor(observation), axis=0),
                                      tf.expand_dims(tf.convert_to_tensor(reward), axis=0))

            action = policy.action(time_step).action
            observation, reward, done, info = env.step(action.numpy()[0])

            reward = np.float32(reward)

            if doneAfterEnd and done:
                print(f"Episode finished after {i + 1} timestep.")
                break
    if closeEnv:
        env.close()


def renderEnvModel(envName="ActorCritic", model: keras.models = None, steps=200,
                   epizodes=1):
    env = gym.make(envName)
    state = tf.constant(env.reset(), dtype=tf.float32)

    for i in range(steps):
        env.render()

        state = tf.expand_dims(state, 0)
        action_probs, _ = model(state)
        action = np.argmax(action_probs.numpy())

        state, reward, done, _ = env.step(action)
        if done:
            break

    env.close()


def viewEnv(envName="CartPole-v0", steps=500):
    env = gym.make(envName)
    env.reset()

    for i in range(steps):
        env.render()
        _, reward, done, _ = env.step(env.action_space.sample())
        if done:
            print(f"Zakończono po {i} krokach")
            break

    env.close()


def printEnv(env):
    print(env.action_space)
    print(env.observation_space)

    print(env.observation_space.high)
    print(env.observation_space.low)


def printEnvNames():
    envsList = envs.registry.all()
    for i in envsList:
        print(i.id)


def loadAgent(agentDir: str, savedPolicy: str):
    return tf.saved_model.load(os.path.join(agentDir, savedPolicy))


def loadModel(modelDir: str, modelName: str) -> tf.keras.Model:
    return keras.models.load_model(os.path.join(modelDir, modelName))


if __name__ == "__main__":
    cartpole = "CartPole-v1"
    lunar_lander = "LunarLander-v2"
    montezuma = "MontezumaRevenge-ram-v0"

    envName = lunar_lander
    # renderEnv(envName, loadAgent(agentDir, envName))

    # renderEnvModel(cartpole, loadModel(modelsDir, envName))

    viewEnv(lunar_lander)

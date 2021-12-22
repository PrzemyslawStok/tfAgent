import gym
from gym import envs

from gym import spaces

import tensorflow as tf
import os

import tensorflow_probability

agentDir = "savedAgents"

def renderEnv(envName="CartPole-v0", doneAfterEnd=True, closeEnv=False, steps=100, epizodes=1):
    env = gym.make(envName)
    env1 = env.unwrapped

    for episode in range(epizodes):
        env.reset()

        for i in range(steps):
            env.render()
            action = env.action_space.sample()
            _, _, done, _ = env.step(action)
            if doneAfterEnd and done:
                print(f"Episode finished after {i + 1} timestep.")
                break
    if closeEnv:
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

def loadAgent(agentDir: str, savedPolicy:str):
    return tf.saved_model.load(os.path.join(agentDir,savedPolicy))

if __name__ == "__main__":
    agent = loadAgent(agentDir,"Cartpole-v1-trained-agent")
    renderEnv("LunarLander-v2")
    printEnvNames()

    space = spaces.Discrete(10)

    print(space)

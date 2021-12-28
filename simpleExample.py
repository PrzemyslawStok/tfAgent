import numpy as np
import random
import matplotlib.pyplot as plot

import timeit


def reward(prob: float) -> float:
    reward = 0
    for i in range(10):
        if random.random() < prob:
            reward += 1
    return reward


# greedy method to select best arm based on memory array
def bestArm(a):
    bestArm = 0  # default to 0
    bestMean = 0
    for u in a:
        avg = np.mean(a[np.where(a[:, 0] == u[0])][:, 1])  # calculate mean reward for each action
        if bestMean < avg:
            bestMean = avg
            bestArm = u[0]
    return bestArm


def bestArm1(experience: np.ndarray) -> int:
    bestArm = 0
    bestMean = 0
    usedArms = np.unique(experience[:, 0])

    for armNo in usedArms:
        armExperience = experience[np.where(experience[:, 0] == armNo)]
        armMean = np.mean(armExperience[:, 1])

        if armMean > bestMean:
            bestMean = armMean
            bestArm = armNo

    return bestArm


np.random.seed(5)

if __name__ == "__main__":
    n = 10
    arms = np.random.rand(n)
    eps = 0.1  # probability of exploration action

    av = np.array([np.random.randint(0, (n + 1)), 0]).reshape(1, 2)  # av = action-value

    plot.xlabel("Number of times played")
    plot.ylabel("Average Reward")

    maxIteration = 500

    startTime = timeit.default_timer()

    scatterData = np.empty([2, maxIteration])
    for i in range(maxIteration):
        if random.random() > eps:  # greedy exploitation action
            choice = bestArm1(av)
            thisAV = np.array([[choice, reward(arms[choice])]])
            av = np.concatenate((av, thisAV), axis=0)
        else:  # exploration action
            choice = np.where(arms == np.random.choice(arms))[0][0]
            thisAV = np.array([[choice, reward(arms[choice])]])  # choice, reward
            av = np.concatenate((av, thisAV), axis=0)  # add to our action-value memory array
        # calculate the mean reward
        runningMean = np.mean(av[:, 1])
        scatterData[0, i] = i
        scatterData[1, i] = runningMean

    plot.scatter(scatterData[0], scatterData[1])

    endTime = timeit.default_timer()

    print(f"elapsed time {(endTime - startTime):0.5f}s")
    plot.show()

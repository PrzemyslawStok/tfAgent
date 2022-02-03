import numpy as np
from tf_agents.environments import py_environment, utils, wrappers, tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from EnvBase import EnvBase


class GraphEnv(py_environment.PyEnvironment):
    def __init__(self, no_of_prim_requirements, no_of_resources, no_of_motors, env_rules,
                 list_of_non_renewable_resources, resource_use_limit, max_iterations, handle_auto_reset=False):
        super().__init__(handle_auto_reset=handle_auto_reset)

        self._envBase = EnvBase(no_of_prim_requirements, no_of_resources, no_of_motors, env_rules,
                                list_of_non_renewable_resources, resource_use_limit, max_iterations)

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(4,), dtype=np.int32, minimum=[0, 0, 0, 0], maximum=[5, 5, 5, 5], name='observation')
        self._state = [0, 0, 5, 5]  # represent the (row, col, frow, fcol) of the player and the finish
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = [0, 0, 5, 5]
        self._episode_ended = False
        return ts.restart(np.array(self._state, dtype=np.int32))

    def move(self, action):
        coordinates = np.random.randint(0, 5)
        self._state = [0, 1, 0, 1]

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        self.move(action)

        if self.game_over():
            self._episode_ended = True

        if self._episode_ended:
            if self.game_over():
                reward = 100
            else:
                reward = 0
            return ts.termination(np.array(self._state, dtype=np.int32), reward)
        else:
            return ts.transition(
                np.array(self._state, dtype=np.int32), reward=0, discount=0.9)

    def game_over(self):
        row, col, frow, fcol = self._state[0], self._state[1], self._state[2], self._state[3]
        return row == frow and col == fcol


if __name__ == '__main__':
    prime_requirements_indexes = [8, 9, 10]

    env_rules = {0: [[3, 3]],
                 1: [[4, 4]],
                 2: [[5, 5]],
                 3: [],
                 4: [[6, 6]],
                 5: [[4, 7]],
                 6: [[7, 8]],
                 7: [[6, 9]],
                 8: [[0, 0]],
                 9: [[1, 1]],
                 10: [[2, 2]],
                 11: []}

    motor_output = ['m_0', 'm_1', 'm_2', 'm_3', 'm_4', 'm_5', 'm_6', 'm_7', 'm_8', 'm_9']

    list_of_non_renewable_resources = [3]

    no_of_prim_requirements = len(prime_requirements_indexes)
    no_of_resources = len(env_rules) - no_of_prim_requirements - 1
    no_of_motors = len(motor_output)

    RESOURCE_USE_LIMIT = np.array([15, 15, 15, 9999999, 15, 15, 15, 15])
    MAX_ITERATIONS = 10000

    env = GraphEnv(no_of_prim_requirements, no_of_resources, no_of_motors, env_rules, list_of_non_renewable_resources,
                   RESOURCE_USE_LIMIT, MAX_ITERATIONS)

    utils.validate_py_environment(env, episodes=5)

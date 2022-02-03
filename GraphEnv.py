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
            shape=(), dtype=np.int32, minimum=0, maximum=self._envBase.getActionSpace() - 1, name='action')

        obeservation_space_length = self._envBase.getObservationSpace()

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(obeservation_space_length,), dtype=np.int32,
            minimum=list(np.zeros([obeservation_space_length], dtype=np.int32)),
            maximum=list(2 * np.ones([obeservation_space_length], dtype=np.int32)), name='observation')

        self._state = self._envBase.reset().astype(dtype=np.int32)

    def sample(self):
        return self._envBase.action_space___sample()

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = self._envBase.reset().astype(dtype=np.int32)
        return ts.restart(np.array(self._state, dtype=np.int32))

    def _step(self, action):
        state, reward, done, iteration = self._envBase.step_probabilistic_resources(action)
        print(f"iteration {iteration}")

        if done:
            return ts.termination(np.array(self._state, dtype=np.int32), reward)
        else:
            return ts.transition(
                np.array(self._state, dtype=np.int32), reward=reward, discount=0.9)


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
    MAX_ITERATIONS = 10

    env = GraphEnv(no_of_prim_requirements, no_of_resources, no_of_motors, env_rules, list_of_non_renewable_resources,
                   RESOURCE_USE_LIMIT, MAX_ITERATIONS)

    state = env.reset()

    for i in range(5):
        state = env.step(env.sample())
        print(state)

    utils.validate_py_environment(env, episodes=5)

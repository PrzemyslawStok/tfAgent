import numpy as np
from tf_agents.environments import py_environment, utils, wrappers, tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

class GraphEnv(py_environment.PyEnvironment):
    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(4,), dtype=np.int32, minimum=[0, 0, 0, 0], maximum=[5, 5, 5, 5], name='observation')
        self._state = [0, 0, 5, 5]  # represent the (row, col, frow, fcol) of the player and the finish
        self._episode_ended = False
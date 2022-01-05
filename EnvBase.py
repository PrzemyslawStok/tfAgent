import numpy as np

class EnvBase:

    def __init__(self, no_of_prim_requirements, no_of_resources, no_of_motors, env_rules,
                 list_of_non_renewable_resources, resource_use_limit, max_iterations):
        self.iteration = 0
        self.max_iterations = max_iterations
        self.episode_finished = False

        self.prim_requirements_thresholds = 1 * np.ones(no_of_prim_requirements)
        self.prim_requirements_die_level = np.zeros(no_of_prim_requirements)

        # prim requirement increase rate
        self.tc = np.ones(no_of_prim_requirements)

        # resource decline rate
        self.resources_rate = np.ones(no_of_resources) * 10

        self.no_of_prim_requirements = no_of_prim_requirements
        self.no_of_motors = no_of_motors
        self.no_of_resources = no_of_resources
        self.env_rules = env_rules

        self.observation_space___n = no_of_prim_requirements + no_of_resources
        self.action_space___n = no_of_motors * no_of_resources + 1  # +1 do noting

        # Resources availabilty by probability calc.
        self.resources_prob = np.ones(no_of_resources)
        self.resources_use_count = np.zeros(no_of_resources, dtype="int")
        self.resources_present = np.ones(no_of_resources)  # True / False for sensors

        # Resource is limited
        self.list_of_non_renewable_resources = list_of_non_renewable_resources
        self.resources_use_limit = np.ones(no_of_resources, dtype="int") * resource_use_limit

        self.prim_requirements = np.zeros(no_of_prim_requirements)

        # [1, , ] - reward for reducing dominating requirement
        # [ ,1, ] - reward for reducing requirement above threshold but requirement is not dominating
        # [ , ,1] - reward for reducing requirement below threshold.

        self.rewards = np.array([1, 1, 0])

        self.was_last_action_useful = False

    def action_space___sample(self):
        random_motor = np.random.randint(0, self.action_space___n - 1)
        return random_motor

    def calc_resource_avaiable(self):
        '''
        self.resources_avaiable=np.random.random(self.no_of_resources) <= self.resources_prob
        '''
        self.resources_present = np.random.random(self.no_of_resources) <= (
                1 / (1 + self.resources_use_count / self.resources_use_limit))

        pass

    def calc_resource_level_discretized(self):
        # resources are limited
        list_to_return = []

        for item in self.calc_resource_level():
            if item > 0:
                list_to_return.append(1)
            else:
                list_to_return.append(0)

        return list_to_return

    def calc_resource_level(self):
        # resources are limited
        return self.resources_use_limit - self.resources_use_count

    def check_if_resources_has_run_out(self):
        resource_uses = self.resources_use_count

        # print(resource_uses)
        # print(self.get_resources_use_limit())

        for item in self.list_of_non_renewable_resources:

            if ((self.resources_use_limit()[item] - resource_uses[item]) == 0):
                self.episode_finished = True

                # if len((np.where((envML.get_resources_use_limit()-envML.get_resource_uses()) == 0))[0]) == 0:
                return self.episode_finished

        return self.prim_requirements

    def get_motor_id(self, action_id, no_of_resources):
        return int(np.floor(action_id / no_of_resources))

    def get_state_discretized_requirements_probabilistic_resources(self):
        tmp_prim_requirements = []

        for i in range(0, len(self.prim_requirements)):
            if self.prim_requirements[i] > self.prim_requirements_thresholds[i]:
                if self.prim_requirements[i] == self.prim_requirements.max():
                    tmp_prim_requirements.append(2)
                else:
                    tmp_prim_requirements.append(1)
            else:
                tmp_prim_requirements.append(0)

        return (tmp_prim_requirements + self.resources_present.tolist())

    def get_state_discretized_resources(self):
        tmp_prim_requirements = []
        for i in range(0, len(self.prim_requirements)):
            if self.prim_requirements[i] > self.prim_requirements_thresholds[i]:
                if self.prim_requirements[i] == self.prim_requirements.max():
                    tmp_prim_requirements.append(2)
                else:
                    tmp_prim_requirements.append(1)
            else:
                tmp_prim_requirements.append(0)

        return (tmp_prim_requirements + self.calc_resource_level_discretized())

    def get_state_limited_resources(self, help=False):
        if help == True:
            print("RETURNS: self.prim_requirements.tolist() + self.calc_resource_level().tolist")
        return np.array(self.prim_requirements.tolist() + self.calc_resource_level().tolist())

    def get_state_requirements_probabilistic_resources(self):
        tmp_prim_requirements = []

        for i in range(0, len(self.prim_requirements)):
            if self.prim_requirements[i] > self.prim_requirements_thresholds[i]:
                if self.prim_requirements[i] == self.prim_requirements.max():
                    tmp_prim_requirements.append(2)
                else:
                    tmp_prim_requirements.append(1)
            else:
                tmp_prim_requirements.append(0)

        return np.array(self.prim_requirements.tolist() + self.resources_present.tolist())

    def get_resource_id(self, action_id, no_of_resources):
        return np.mod(action_id, no_of_resources, dtype=int)

    def get_was_last_action_useful(self):
        return self.was_last_action_useful

    def make_action_limited_resources(self, action_id):

        self.was_last_action_useful = False
        prim_requirement_was_reliefed = False
        prim_requirement_was_above_threshold = False
        prim_requirement_idx = -1

        episode_finished = False

        # the last action is do nothing action:
        if (action_id < (self.action_space___n - 1)):
            # Determine associated resource
            resource = self.get_resource_id(action_id, self.no_of_resources)
            # Determine associated motor output
            motor = self.get_motor_id(action_id, self.no_of_resources)
            ###print("i "+str(self.iteration)+" R: " + str(resource)+ " M:"+str(motor))

            # Episode finished only when limited resurce is depleted and tried to use one more by agent
            if self.check_if_resources_has_run_out() == True:
                episode_finished = True

            # if a resurce can be depleted, deplete it
            if (self.resources_use_limit[resource] - self.resources_use_count[resource]) > 0:
                self.resources_use_count[resource] += 1

                # check if action was useful
                for resource_idx in range(0, (self.no_of_resources + self.no_of_prim_requirements)):
                    for item in self.env_rules[resource_idx]:
                        # if proper pair of resource and action was called
                        if (item[0] == resource) and (item[1] == motor):

                            self.was_last_action_useful = True

                            # check if action does relevie prim requirement
                            if resource_idx in range(self.no_of_resources,
                                                     self.no_of_resources + self.no_of_prim_requirements):
                                prim_requirement_idx = resource_idx - self.no_of_resources

                                if self.prim_requirements[prim_requirement_idx] > self.prim_requirements_thresholds[
                                    prim_requirement_idx]:
                                    prim_requirement_was_above_threshold = True
                                prim_requirement_was_reliefed = True
                                self.prim_requirements[prim_requirement_idx] = 0
                                break
                            # action restores resurce
                            else:
                                self.resources_use_count[resource_idx] = 0
                                break

        return prim_requirement_was_reliefed, prim_requirement_was_above_threshold, prim_requirement_idx, episode_finished

    def make_action_probabilistic_resources(self, action_id):
        self.was_last_action_useful = False
        prim_requirement_was_reliefed = False
        prim_requirement_was_above_threshold = False
        prim_requirement_idx = -1

        episode_finished = False

        # the last action is do nothing action:
        if (action_id < (self.action_space___n - 1)):
            # Determine associated resource
            resource = self.get_resource_id(action_id, self.no_of_resources)
            # Determine associated motor output
            motor = self.get_motor_id(action_id, self.no_of_resources)
            ###print("i "+str(self.iteration)+" R: " + str(resource)+ " M:"+str(motor))

            # Check if particular resource is present
            # if it is, deplete it
            if self.resources_present[resource] == True:
                self.resources_use_count[resource] += 1

                # check if action was useful
                for resource_idx in range(0, (self.no_of_resources + self.no_of_prim_requirements)):
                    for item in self.env_rules[resource_idx]:
                        # if proper pair of resource and action was called
                        if (item[0] == resource) and (item[1] == motor):

                            self.was_last_action_useful = True

                            # check if action does relevie prim requirement
                            if resource_idx in range(self.no_of_resources,
                                                     self.no_of_resources + self.no_of_prim_requirements):
                                prim_requirement_idx = resource_idx - self.no_of_resources

                                if self.prim_requirements[prim_requirement_idx] > self.prim_requirements_thresholds[
                                    prim_requirement_idx]:
                                    prim_requirement_was_above_threshold = True
                                prim_requirement_was_reliefed = True
                                self.prim_requirements[prim_requirement_idx] = 0
                                break
                            # action restores resurce
                            else:
                                self.resources_use_count[resource_idx] = 0
                                break

        self.calc_resource_avaiable()

        return prim_requirement_was_reliefed, prim_requirement_was_above_threshold, prim_requirement_idx, episode_finished

    def step_limited_resources(self, action):
        self.iteration += 1
        self.prim_requirements += self.tc
        reward = 0

        prim_requirements_last_step = self.prim_requirements.copy()

        prim_requirement_was_reliefed, prim_requirement_was_above_threshold, prim_requirement_idx, episode_finished = self.make_action_limited_resources(
            action)

        self.episode_finished = episode_finished

        if prim_requirement_was_reliefed == True:
            # if (np.sum(self.prim_requirements) < np.sum(prim_requirements_last_step)):
            if (prim_requirements_last_step).max() == prim_requirements_last_step[prim_requirement_idx]:
                # biggest prim requirement was reliefed therefore REWARD is BIGGER
                reward = self.rewards[0]  # self.tc[prim_requirement_idx]
            else:
                reward = self.rewards[1]  # 0.5*self.tc[prim_requirement_idx]

            if prim_requirement_was_above_threshold == False:
                reward = self.rewards[2]
        else:
            for i in range(0, self.prim_requirements.shape[0]):
                # add reward for not played episodes
                reward = 0  # - 1 * self.prim_requirements[i]
                # reward = -2*self.tc[prim_requirement_idx]

        state = np.array(self.get_state_discretized_resources())

        if self.max_iterations == self.iteration:
            self.episode_finished = True

        # if len(self.prim_requirements_die_level[self.prim_requirements_die_level>0])>0:
        if np.any(self.prim_requirements_die_level) > 0:
            prequirements_compare = (self.prim_requirements == self.prim_requirements_die_level)
            if np.any(prequirements_compare):
                # one of the prim requirements reached max level, agent dies
                self.episode_finished = True

                # 3x state variables and reward
        return state, reward, self.episode_finished, self.iteration

    def step_probabilistic_resources(self, action):
        self.iteration += 1
        self.prim_requirements += self.tc
        reward = 0

        prim_requirements_last_step = self.prim_requirements.copy()

        prim_requirement_was_reliefed, prim_requirement_was_above_threshold, prim_requirement_idx, episode_finished = self.make_action_probabilistic_resources(
            action)

        # here always False for probabilistic resources
        self.episode_finished = episode_finished

        if prim_requirement_was_reliefed == True:
            # if (np.sum(self.prim_requirements) < np.sum(prim_requirements_last_step)):
            if (prim_requirements_last_step).max() == prim_requirements_last_step[prim_requirement_idx]:
                # biggest prim requirement was reliefed therefore REWARD is BIGGER
                reward = self.rewards[0]  # self.tc[prim_requirement_idx]
            else:
                reward = self.rewards[1]  # 0.5*self.tc[prim_requirement_idx]

            if prim_requirement_was_above_threshold == False:
                reward = self.rewards[2]
        else:
            for i in range(0, self.prim_requirements.shape[0]):
                # add reward for not played episodes
                reward = 0  # - 1 * self.prim_requirements[i]
                # reward = -2*self.tc[prim_requirement_idx]

        state = np.array(self.get_state_discretized_requirements_probabilistic_resources())

        if self.max_iterations == self.iteration:
            self.episode_finished = True

        # if len(self.prim_requirements_die_level[self.prim_requirements_die_level>0])>0:
        if np.any(self.prim_requirements_die_level) > 0:
            prequirements_compare = (self.prim_requirements == self.prim_requirements_die_level)
            if np.any(prequirements_compare):
                # one of the prim requirements reached max level, agent dies
                self.episode_finished = True

                # 3x state variables and reward
        return state, reward, self.episode_finished, self.iteration

    def render(self):
        pass

    def reset(self):
        self.iteration = 0
        self.episode_finished = False
        self.resources_use_count = np.zeros(self.no_of_resources)
        self.prim_requirements = np.zeros(self.no_of_prim_requirements)

        self.calc_resource_avaiable()

        # returns state
        return np.array(self.prim_requirements.tolist() + self.resources_present.tolist())

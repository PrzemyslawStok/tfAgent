MAX_ITERATIONS = 5000
RESOURCE_USE_LIMIT = 3


class EnvML:
    ACTION_SPACE_SIZE = 0

    def __init__(self, no_of_prim_requirementss, no_of_resources, no_of_motors, env_rules,
                 list_of_non_renewable_resources):
        self.iteration = 0
        self.max_iterations = MAX_ITERATIONS
        self.episode_finished = False

        # prim requirements increase rate
        self.tc = 0.1
        # resource decline rate
        self.resources_rate = np.ones(no_of_resources) * 10

        self.no_of_prim_requirementss = no_of_prim_requirementss
        self.no_of_motors = no_of_motors
        self.no_of_resources = no_of_resources
        self.env_rules = env_rules

        self.observation_space___n = no_of_prim_requirementss + no_of_resources
        self.action_space___n = no_of_motors * no_of_resources
        self.ACTION_SPACE_SIZE = self.action_space___n

        # Resources availabilty by probability calc.
        self.resources_prob = np.ones(no_of_resources)
        self.resources_use_count = np.zeros(no_of_resources)
        self.resources_avaiable = np.ones(no_of_resources)  # True / False for sensors

        # Resuorce is limited
        self.list_of_non_renewable_resources = list_of_non_renewable_resources
        self.resources_use_limit = np.ones(no_of_resources) * RESOURCE_USE_LIMIT

        self.prim_requirementss = np.zeros(no_of_prim_requirementss)

        self.was_last_action_useful = False
        pass

    def action_space___sample(self):
        random_motor = randint(0, self.action_space___n - 1)
        return random_motor

    def step_limited_resources(self, action):
        self.iteration += 1
        self.prim_requirementss += self.tc

        prim_requirementss_last_step = self.prim_requirementss.copy()

        prim_requirements_was_reliefed, prim_requirements_idx = self.make_action_limited_resources(action)

        if prim_requirements_was_reliefed == True:
            # if (np.sum(self.prim_requirementss) < np.sum(prim_requirementss_last_step)):
            if (prim_requirementss_last_step).max() == prim_requirementss_last_step[prim_requirements_idx]:
                # the biggest prim requirements was reliefed
                # reward = 2*self.tc - self.tc
                reward = 2
            else:
                # biggest prim requirements was reliefed therefore REWARD is BIGGER
                reward = 1
        else:
            # reward = -2*self.tc
            reward = 0

        state = self.get_state_limited_resources()

        if self.check_resources_run_out() == True:
            self.episode_finished = True
            # reward -= (self.max_iterations-self.iteration) * 2 * self.tc
            reward = 0
            finished = True
        elif self.max_iterations == self.iteration:
            finished = True
            self.episode_finished = True
        elif self.episode_finished == True:
            finished = True
        else:
            finished = False
        # 3x state variables and reward
        return state, reward, finished

    def step_prob(self, action):
        self.iteration += 1
        self.prim_requirementss += self.tc

        prim_requirementss_last_step = self.prim_requirementss.copy()

        self.make_action(action)

        reward = np.sum(self.prim_requirementss) - np.sum(prim_requirementss_last_step)

        state = self.get_state()

        if self.max_iterations == self.iteration:
            finished = True
        else:
            finished = False
        # 3x state variables and reward
        return state, reward, finished

    def get_state_limited_resources(self):
        return np.array(self.prim_requirementss.tolist() + self.calc_resource_level().tolist())

    def get_state_probabilistic_resources(self):
        return np.array(
            self.prim_requirementss.tolist() + self.calc_resource_avaiable().tolist() + self.calc_resource_probabilities().tolist())

    def calc_resource_probabilities(self):
        self.resources_prob = 1 / (1 + (self.resources_use_count / self.resources_rate))
        return self.resources_prob

    def calc_resource_avaiable(self):
        self.resources_avaiable = np.random.random(self.no_of_resources) <= self.resources_prob
        return self.resources_avaiable

    def calc_resource_level(self):
        # resources are limited
        return self.resources_use_limit - self.resources_use_count

    def make_action_probabilistic_resources(self, action_id):
        self.env_rules

        # Determine associated resource
        resource = get_resource_id(action_id, self.no_of_resources)

        # Determine associated motor output
        motor = get_motor_id(action_id / self.no_of_resources)
        ###print("i "+str(self.iteration)+" R: " + str(resource)+ " M:"+str(motor))

        self.was_last_action_useful = False

        if self.resources_avaiable[resource] == 1:
            self.resources_use_count[resource] += 1

            for resource_idx in range(0, self.no_of_resources + self.no_of_prim_requirementss):
                for item in self.env_rules[resource_idx]:
                    if (item[0] == resource) and (item[1] == motor):

                        self.was_last_action_useful = True
                        # if resource relieves prim requirements
                        if resource_idx in range(self.no_of_resources,
                                                 self.no_of_resources + self.no_of_prim_requirementss):
                            self.prim_requirementss[self.no_of_resources - resource_idx] = 0
                            break
                        else:
                            self.resources_use_count[resource_idx] = 0
                            break

    def make_action_limited_resources(self, action_id):
        # Determine associated resource
        resource = np.mod(action_id, self.no_of_resources, dtype=int)
        # Determine associated motor output
        motor = int(np.floor(action_id / self.no_of_resources))
        ###print("i "+str(self.iteration)+" R: " + str(resource)+ " M:"+str(motor))

        self.was_last_action_useful = False

        prim_requirements_was_reliefed = False
        prim_requirements_idx = -1

        if (self.resources_use_limit[resource] - self.resources_use_count[resource]) > 0:
            self.resources_use_count[resource] += 1

            for resource_idx in range(0, self.no_of_resources + self.no_of_prim_requirementss):
                for item in self.env_rules[resource_idx]:
                    if (item[0] == resource) and (item[1] == motor):

                        self.was_last_action_useful = True
                        # if resource relieves prim requirements
                        if resource_idx in range(self.no_of_resources,
                                                 self.no_of_resources + self.no_of_prim_requirementss):
                            self.prim_requirementss[self.no_of_resources - resource_idx] = 0
                            prim_requirements_was_reliefed = True
                            prim_requirements_idx = self.no_of_resources - resource_idx
                            break
                        # if relieves
                        else:
                            self.resources_use_count[resource_idx] = 0
                            break

        return prim_requirements_was_reliefed, prim_requirements_idx

    def get_motor_id(self, action_id, no_of_resources):
        return int(np.floor(action_id / no_of_resources))

    def get_resource_id(self, action_id, no_of_resources):
        return np.mod(action_id, no_of_resources, dtype=int)

    def get_was_last_action_useful(self):
        return self.was_last_action_useful

    def render(self):
        pass

    def reset_resources_probabilistic(self):
        self.iteration = 0
        self.episode_finished = False

        self.resources_rate = np.ones(no_of_resources) * 10
        self.resources_prob = np.ones(no_of_resources)
        self.resources_use_count = np.zeros(no_of_resources)
        self.resources_avaiable = np.ones(no_of_resources)

        self.prim_requirementss = np.zeros(no_of_prim_requirementss)

        state = np.array(
            self.prim_requirementss.tolist() + self.calc_resource_avaiable().tolist() + self.calc_resource_probabilities().tolist())

        # 3x state variables and reward
        return state

    def reset_resources_limited(self):
        self.iteration = 0
        self.episode_finished = False
        self.resources_use_count = np.zeros(no_of_resources)
        self.resources_use_limit = np.ones(no_of_resources) * RESOURCE_USE_LIMIT

        self.prim_requirementss = np.zeros(no_of_prim_requirementss)

        state = np.array(self.prim_requirementss.tolist() + self.calc_resource_level().tolist())

        # 3x state variables and reward
        return state

    def get_resource_uses(self):
        return self.resources_use_count

    def get_resources_use_limit(self):
        return self.resources_use_limit

    def check_resources_run_out(self):
        resource_uses = self.get_resource_uses()

        for item in list_of_non_renewable_resources:
            if (resource_uses[item] - self.get_resources_use_limit()[item]) == 0:
                self.episode_finished = True
                return True

        # if len((np.where((envML.get_resources_use_limit()-envML.get_resource_uses()) == 0))[0]) == 0:
        return False

    def get_prim_requirementss(self):
        return self.prim_requirementss

# envML = EnvML(2,2,4)

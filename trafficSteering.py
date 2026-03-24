# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import gymnasium as gym # is required for RL implementations
import numpy as np # handles matrices and vectors

class TraffciSteeringEnv(gym.Env): # class that defines RL enviornment for traffic steering use case
    # TODO: Add extra parameters inside the traffic steering env class to have outside users change internal param
    def __init__(self):
        """ Method that initializes the environment
        TODO: Change the parameters in scale such as num UEs, BS to follow diagram in paper & other parameters to see
        influence on results. Also see size of network maybe smaller and larger and see effets
        """
        self._size_of_network = 1000 # Max distance a box of 1000x1000 meters
        self._num_gnbs = 4
        self._num_ue = 5
        self._max_steps = 200
        self._ho_cost = 0.3
        self._frequency = 3.5e9 # carrier frequency c band in Hz
        self._power_transmitted = 44 # power transmitted by BS in dBm
        self._gain_transmitted = 8 # antenna gain of BS in dBi
        self._gain_received = 5.3 # antenna gain of UE in dBi
        self._bandwidth = 20e6 # bandwidth in Hz
        self._temperature = 290 # ambient temp in kelvin
        self._noise_figure = 7 # noise figure in dB for UE


        super().__init__() # initialize the parent class

        # Observation Space
        """We will have a value per UE expressing the QoS and a value per BS/gNB to express capacity/load"""
        # Need a min and max and there will no dtype as it is mixed floats and integers
        throughputMinPerUe = np.zeros(self._num_ue)
        throughputMaxPerUe = np.full(self._num_ue, 800) # max value calculated by matlab script

        # same for BS value
        minLoad = np.zeros(self._num_gnbs)
        maxLoad = np.full(self._num_gnbs, self._num_ue)

        # Combining in a single variable
        obsLow = np.concatenate([throughputMinPerUe, minLoad])
        obsHigh = np.concatenate([throughputMaxPerUe, maxLoad])

        self.observation_space = gym.spaces.Box(low=obsLow, high=obsHigh)

        # Action Space
        possibleActions = [self._num_gnbs + 1] * self._num_ue # steer to all possible gnbs + 1 (NOOP) and that for all UEs
        self.action_space = gym.spaces.MultiDiscrete(possibleActions)

        # Values used for calculations and decision making
        # None is used to force use of reset which initializes them before use of step or other methods.
        self._throughput = None # (NUM_UE, NUM_BS) a throughput value (Mbps) per UE for each BS
        self._assignment = None # (NUM_UE) assignment/BS number for each UE
        self._ue_pos = None # (NUM_UE, 2) 2D position for each UE in meters
        self._step_count = 0 # A number to keep track/simulate time. Works with self_max_steps
        self._distance = None# (NUM_UE, NUM_BS) distance in meters for each scenario. Needed for display purposes
        self._gnb_pos = np.array([
            [250, 250],
            [250, 750],
            [750, 250],
            [750, 750],
        ]) # 4 BSs stationed symmetrically and equidistant
        """ TODO: May need to generalize given any number find equi-distant coordinates """

        return

    def compute_throughput(self):
        """
        Method that computes the throughput between all UEs and all gNBs used for decisions and display purposes
        Many TODOs to improve validity
        TODO: Add load in decision making. E.g divide throughput by number of UEs
        """
        episilon = 1e-6 # for stability and avoiding to divide by 0

        for ue in range(self._num_ue):
            for gnb in range(self._num_gnbs):
                # compute distance
                currentDistance = np.linalg.norm(self._ue_pos[ue] - self._gnb_pos[gnb]) + episilon
                self._distance[ue,gnb] = currentDistance # Varaible to now be used

                # free space pathloss
                pathloss = -147.6 + 20*np.log10(currentDistance) + 20*np.log10(self._frequency)

                # Received Power on UE in dBm (Friis Equation)
                powerReceived = self._power_transmitted + self._gain_transmitted + self._gain_received - pathloss

                # Convert to dB
                powerReceived = powerReceived - 30

                # Noise Power on UE in dB
                k = -228.6 # Bolstman constant in dBW/dB
                powerNoise = k + 10*np.log10(self._temperature)+10*np.log10(self._bandwidth)+self._noise_figure

                # SNR (Signal to Noise ratio) in dB
                snr = powerReceived - powerNoise

                # SNR in linear scale
                snr = 10**(snr/10)

                # get throughput in Mega bits per second using shannon capacity theorem
                self._throughput[ue, gnb] = self._bandwidth * np.log2(1+snr)/(1e6) # variable to now be used
        return

    def compute_ues_per_cell(self, assignment):
        """
        :param assignment: A matrix of size (numUEs, 2) to map each UE to a cell/BS
        :return: vector (numGNBs, 1) in a numpy array which will return the number of UEs for each cell.
        """
        mapAssignment = np.zeros(self._num_gnbs)
        for gnb in range(self._num_gnbs):
            mapAssignment[gnb] = np.sum(assignment == gnb)
        return mapAssignment


    def get_observation(self):
        """
        A methods that returns the observation of the agent which will be the state of the environment
        :return: A vector of size (numUEs + numBS, 1) Making it UE and BS centric. Unlike paper which is UE centric
        """
        return np.concatenate([self._throughput[range(self._num_ue), self._assignment], self.compute_ues_per_cell(self._assignment)])


    def compute_reward_function(self, oldThroughput, newThroughput, oldAssignment, newAssignment):
        """
        Computes the reward of the function by taking a log of the throughputs and adding a penaly for lambda

        positive if throughput improves
        zero if unchanged
        TODO: Examine if we should encourage unchange
        Negative if HO penalty exceeds throughput benefit

        TODO: Change the HO cost to a varaiable on such as exponent RV (exp decay)

        TODO: Add UE weight similar to paper for maybe diff 5G use case. E.G emmb wants greater throughput

        TODO: Add a more BS centric approach as well take in consideration load

        R_t = sum_over_ues[log2(curr_throughput)]
        :param oldThroughput: the throughtput from the original setup of that step per UE per BS (numUE, numBS)
        :param newThroughput: the throughtput from the updated setup of that step per UE per BS (numUE, numBS)
        :param oldAssignment: the map matrix from the original setup of that step per BS (numBS, 1)
        :param newAssignment: the map matrix from the updated setup of that step per BS (numBS, 1)
        :return: reward & number of handovers for info purposes
        """
        numHO = 0
        rNew = 0.0
        #rOld = 0.0
        epsilon = 1e-6

        # Compute reward per UE and aggregrate & get num HO
        for ue in range(self._num_ue):
            rNew += np.log2(newThroughput[ue] + epsilon)
            #rOld += np.log2(oldThroughput[ue] + epsilon)
            if newAssignment[ue] != oldAssignment[ue]: # there is a HO as it changed
                numHO += 1
        
        #return rNew - rOld - numHO*self._ho_cost, numHO
        return rNew- numHO*self._ho_cost, numHO


    def reset(self, seed=None):
        """
        Method that resets the environment and establishes a starting point.
        :param seed an integer for reproducibility
        :return:
        obs which is a vector of size (numUEs + numBS, 1) Making it UE and BS centric. Unlike paper which is UE centric
        info a dictionary that hold a summary of important information. The format is name(str) and value (usally a list)
        """
        super().reset(seed=seed)

        self._step_count = 0
        self._throughput = np.zeros((self._num_ue, self._num_gnbs))
        self._distance = np.zeros((self._num_ue, self._num_gnbs))

        # Starting distribution use a uniform distribution accross grid to place UEs
        # TODO: add a Poisson distribution to add random entrance and exit of UEs
        self._ue_pos = self.np_random.uniform(low=0, high=self._size_of_network, size=(self._num_ue, 2))

        # Set Random Assignments
        # TODO: Try to have gnbs numbered from 1-self._num_gnbs not from 0
        self._assignment = self.np_random.integers(low=0, high=self._num_gnbs, size=self._num_ue)

        # compute throughput
        self.compute_throughput() # assigns self.throughput + self.distance

        # Set step number to 0
        self._step_count = 0

        # get observations which is essentially current state
        state = self.get_observation()

        # Info to be used for display or key decisions. Implemented as a dictionary kind of like a database
        info = \
        {
            "ue_positions" : self._ue_pos.tolist(),
            "gnb_positions": self._gnb_pos.tolist(),
            "assignment"   : self._assignment.tolist(),
            "throughput"   : self._throughput.tolist(),
            "distance"     : self._distance.tolist(),
            "num_handovers": 0
        }
        # most implements are implemented using a list

        return state, info

    def step(self, action):
        """
        Method that tales a step in the environment given an action.
        :param action: A vector of size (numUEs,1) which highlights to which BS the UE will be steered too.
        gnb + 1 = NOP
        :return:
        obs which is a vector of size (numUEs + numBS, 1) Making it UE and BS centric. Unlike paper which is UE centric
        reward a numerical value to show success
        terminated a boolean showing whether the episode ended or not
        truncated a boolean showing whether the episode ended unexpectedly
        info a dictionary that hold a summary of important information. The format is name(str) and value (usally a list)
        """

        # initialize action var & make sure valid in case RL agent sends something wrong
        action = np.array(action)
        print(f"action: {action}")
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # store prev state (better to use copy as numpy arrays are mutable and hence can be modified by this mutation)
        oldAssignment = self._assignment.copy()
        oldThroughput = self._throughput[range(self._num_ue), oldAssignment].copy()

        # take action by changing assignment
        newAssignment = oldAssignment.copy()
        for ue in range(self._num_ue):
            if action[ue] != self._num_gnbs:
                newAssignment[ue] = action[ue]
        self._assignment = newAssignment # action effectively taken

        # Simulate Ue mouvement random walk as per NS3 simulator
        # TODO: look if random walk is the best as it goes in arb dir
        # TODO: Try to match 3GPP standards speed

        ueMobility = self.np_random.normal(0, 10, size=self._ue_pos.shape)

        # makes sure it can not exceed boundary
        self._ue_pos = np.clip(self._ue_pos + ueMobility, 0, self._size_of_network)

        # recompute metrics
        self.compute_throughput()

        # get reward
        #print(self._throughput[range(self._num_ue), self._assignment])
        reward, numHO = self.compute_reward_function(oldThroughput, self._throughput[range(self._num_ue), self._assignment]
                                                     , oldAssignment, newAssignment)

        # get state obs
        obs = self.get_observation()
        self._step_count += 1

        # never modify termination as it technically never ends only truncated
        terminated = False
        if self._step_count >= self._max_steps:
            truncated = True
        else:
            truncated = False


        # update info
        info = \
        {
            "ue_positions": self._ue_pos.tolist(),
            "gnb_positions": self._gnb_pos.tolist(),
            "assignment": self._assignment.tolist(),
            "throughput": self._throughput.tolist(),
            "distance"  : self._distance.tolist(),
            "num_handovers": numHO
        }

        return obs, reward, terminated, truncated, info


    def render(self):
        """
        Prints the key information to show the environment. Key information displayed in the terminal
        """

        uePerCell = self.compute_ues_per_cell(self._assignment)

        terminal = []

        upperBar = "\n" + "=" * 60
        terminal.append(upperBar)

        description = f" O-RAN Traffic Steering | Step {self._step_count} out of {self._max_steps}"
        terminal.append(description)

        lowerBar = "=" * 60
        terminal.append(lowerBar)

        tableHeader = f"\n {'UE':>3} {'Cell':>5} {'Throughput (Mbps)':>18}"
        terminal.append(tableHeader)

        # TODO: To be generilized for a greater network
        tableHeaderContinued = f"{'gNB0 (m)':>9} {'gNB1 (m)':>9} {'gNB2 (m)':>9} {'gNB3 (m)':>9}"
        terminal.append(tableHeaderContinued)

        headerSeperation = "-" * 50
        terminal.append(headerSeperation)

        # print information from each step
        for ue in range(self._num_ue):
            gnb = self._assignment[ue]
            throughput = self._throughput[ue, gnb]
            # TODO: Generalize below to have multiple gnbs
            distanceToGnb0 = self._distance[ue, 0]
            distanceToGnb1 = self._distance[ue, 1]
            distanceToGnb2 = self._distance[ue, 2]
            distanceToGnb3 = self._distance[ue, 3]

            # TODO: Generalize below to have multiple gnbs
            fillTableInfo = (f"{ue:>3} gNB-{gnb:>1} {throughput:>17.2f} {distanceToGnb0:>9.1f} {distanceToGnb1:>9.1f} "
                             f"{distanceToGnb2:>9.1f} {distanceToGnb3:>9.1f}")
            terminal.append(fillTableInfo)


            """ Second table for BS centric """
            secondTableBSCentricHeader = f"\n {'Cell':>5} {'UEs':>5}"
            terminal.append(secondTableBSCentricHeader)

            header2Seperation = "-" * 15
            terminal.append(header2Seperation)

            # print info for second table gnb centric
            for gnb in range(self._num_gnbs):
                numUes = int(uePerCell[gnb])
                fillTable2Info = f" gNB-{gnb} {numUes:>5}"
                terminal.append(fillTable2Info)

            output = "\n".join(terminal) + "\n"
            print(output)
            return output


    def close(self):
        """
        Closes the environment
        """
        return # done


    """     END OF CLASS TRAFFIC STEERING       """





def greedy_distance_policy(env):
    """ Will return the action vector that will steer the UE to closest BS which is a baseline policy
    Note: env is sent instead of self as it is meant to be an external method (agent)
    :param env variable representing the traffic steering environemnt.
    :return: the action that it to be taken if only the distance is considered. Will act as a baseline
    """
    action = np.zeros(env._num_ue, dtype=int)

    for ue in range(env._num_ue):
        closestGnb = np.argmin(env._distance[ue,:])
        if closestGnb == env._assignment[ue]:
            action[ue] = env._num_gnbs # NOP
        else:
            action[ue] = closestGnb

    return action

def getNumGnbs(env):
    """
    Get method for number of gnbs
    :return: integer representing the number of base stations in environment
    """
    return env._num_gnbs

def getNumUe(env):
    """
    Get method for number of ues
    :return: integer representing the number of ues in environment
    """
    return env._num_ue



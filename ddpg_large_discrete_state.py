"""
this example illustrates how use the DDPG (deep determined policy gradient) to solve the large state space and large
discrete action space problem.

The environment is designed by some modules functions which user can define the component numbers, corresponding
state transition matrix, observation matrix, action space.

We should emphasize the action defined part. Normally, the maintenance actions in infrastructural management is
discrete. If a component has four available actions, for multiple-component structure, the total action combinations is
4^N (N is number of component). To avoid the curse of dimensionality, the discrete action space is evolved to continuous
action domain. The output of layer is number of component.

DDPG is developed from DQN and actor-critic method.

copyright- Lai Li (The Hong Kong Polytechnic University, main programmer)
Wang Aijun (over 140 points in the postgraduate entrance mathematical exams, second programmer)
Prof. Dong You (The Hong Kong Polytechnic University, guidance)
"""


import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import  tensorflow as tf
from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from tensorflow import keras
import numpy as np
import scipy.stats as stats

def state_transition(state_number,prob):
    """ Markov model state transition matrix """
    transition_matrix = np.zeros((1, state_number, state_number))
    transition_matrix[0, state_number-1, state_number-1] = 1

    for step, ri in enumerate(prob):
        Pi = ri / (1 + ri)
        Qi = 1 / (1 + ri)
        transition_matrix[0,step, step] = Pi
        transition_matrix[0,step, step + 1] = Qi
    return transition_matrix

def observation(accuracy,state_number):
    """
    the observation matrix is manually designed, based on the state space
    Args:
        accuracy: the accurate level of structural health monitoring system
        state_number: the state number of a component

    Returns:
        observation_matrix
    """
    observation_matrix = np.zeros((state_number, state_number))
    observation_matrix[0, 0] = accuracy
    observation_matrix[0, 1] = 1 - accuracy
    observation_matrix[state_number - 1, state_number - 2] = 1 - accuracy
    observation_matrix[state_number - 1, state_number - 1] = accuracy
    for step in range(state_number-2):
        observation_matrix[step + 1, step]= (1 - accuracy)/2
        observation_matrix[step + 1, step + 1] = accuracy
        observation_matrix[step + 1, step + 2] = (1 - accuracy) / 2

    return observation_matrix

def action_matrix(action_index,state_number):
    """
    based on the utility value, the
    Args:
        action_index: 0 means Do nothing, 1 means routine maintenance, 2 means rehabilitation, 3 means reinforcement,
        4 means replace
        state_number: the state number of a component
        component_number: the number of component to choose the transition_matrix
        component_state: current component state, is a vector
    Returns:
        repair matrix when agent execute the maintenance action with utility value
    """
    repair_matrix = np.zeros((state_number, state_number))
    if action_index == 0:
        repair_matrix = np.eye(state_number,dtype=int)
    elif action_index == 1:
        repair_matrix[state_number - 1, state_number - 1] = 1
        for i in range(state_number - 1):
            repair_matrix[i, i] = 0.8
            repair_matrix[i, i + 1] = 0.2
    elif action_index == 2:
        repair_matrix[state_number - 1, state_number - 1] = 1
        for i in range(state_number - 1):
            repair_matrix[i, i] = 0.9
            repair_matrix[i, i + 1] = 0.1
    elif action_index == 3:
        repair_matrix[0, 0] = 1
        for i in range(state_number - 1):
            repair_matrix[i + 1, i] = 0.1
            repair_matrix[i + 1, i + 1] = 0.9
    elif action_index == 4:
        repair_matrix[:, 0] = 0.9
        repair_matrix[:, 1] = 0.1

    return repair_matrix

def reward(component_state, action_index, state_number):
    # Reward table is the immediate reward based on the states (column) and actions (row)
    Reward_table = np.zeros((5, state_number))
    # the punishment for the state deterioration
    state_deterioration_punishment = 1.05
    Do_nothing = 0
    routine_maintenance = -1
    repair = -4
    rehabilitation = -9
    replace = -10
    punishment = -30
    cost = np.array([Do_nothing, routine_maintenance, repair, rehabilitation, replace])
    for i in range(5):
        for j in range(state_number):
            Reward_table[i,j] = cost[i] * state_deterioration_punishment ** j
    Reward_table[:, state_number - 1] = punishment

    reward = 0
    for i in range(len(component_state)):
        reward += Reward_table[action_index, i] * component_state[i]
    return reward

class environment():
    """this part define the bridge degradation process"""
    def __init__(self):
        """fundamental parameters of structure or bridge"""
        self.component_number = 16
        self.state_number = 10
        self.accuracy = 0.9

        self.prob1 = np.array([2, 2, 2, 2, 3, 3, 4, 4, 4])
        self.prob2 = np.array([2.5, 2.5, 2.5, 2.5, 3, 3, 3, 3, 4])
        self.prob3 = np.array([2, 2, 2.5, 2.5, 3, 3, 3, 3.5, 3.5])
        self.prob4 = np.array([2, 2, 2, 2.5, 2.5, 2.5, 3, 3, 3])

        self.transition_matrix = np.zeros((self.component_number,self.state_number,self.state_number))

        for i in range(self.component_number):
            if i < 4:
                self.transition_matrix[i, :, :] = state_transition(self.state_number,
                                                                   self.prob1.reshape(self.state_number - 1, 1))
            elif i > 3 and i < 8:
                self.transition_matrix[i, :, :] = state_transition(self.state_number,
                                                                   self.prob2.reshape(self.state_number - 1, 1))
            elif i > 7 and i < 12:
                self.transition_matrix[i, :, :] = state_transition(self.state_number,
                                                                   self.prob3.reshape(self.state_number - 1, 1))
            elif i > 11 and i < 16:
                self.transition_matrix[i, :, :] = state_transition(self.state_number,
                                                                   self.prob4.reshape(self.state_number - 1, 1))
        self.Observation = observation(self.accuracy, self.state_number)

    def reset(self):
        self.state = np.zeros((self.component_number, self.state_number))
        self.state[:, 0] = 0.9
        self.state[:, 1] = 0.1
        self.state = self.state.reshape(-1, self.component_number * self.state_number)
        return self.state

    def step(self, states, action_index, hidden_state):
        state = np.reshape(states, [self.component_number, -1])
        new_hidden_state = np.zeros((1, self.component_number), dtype=int)

        state_update = np.zeros((self.component_number, self.state_number))
        for i in range(self.component_number):
            repair_matrix = action_matrix(action_index[i], self.state_number)
            if action_index[i] == 0:
                transition_matrix = repair_matrix @ self.transition_matrix[i, :, :]
                state_update[i, :] = state[i, :].T @ transition_matrix
            else:
                transition_matrix = repair_matrix
                state_update[i, :] = state[i, :].T @ transition_matrix

            # define hidden state transition
            state_mark = 0.
            Random_number = np.random.uniform(0, 1)
            for j in range(self.state_number):
                state_mark = state_mark + transition_matrix[hidden_state[0, i], j]
                if Random_number <= state_mark:
                    new_hidden_state[0, i] = j
                    break

        # observation part to define current belief state
        Observation_state = np.zeros((self.component_number,), dtype=int)
        observation_matrix = self.Observation
        for i in range(self.component_number):
            obser_mark = 0.
            Random_number = np.random.uniform(0, 1)
            for j in range(self.state_number):
                obser_mark = obser_mark + observation_matrix[new_hidden_state[0, i], j]
                if Random_number <= obser_mark:
                    Observation_state[i] = j
                    break

        # obtain final belief state of components
        belief_state = np.zeros((self.component_number, self.state_number))
        for i in range(self.component_number):
            belief_state[i, :] = state_update[i, :] * observation_matrix[:, Observation_state[i]]
            belief_state[i, :] = belief_state[i, :] / np.sum(belief_state[i,:])

        # immediate reward part r
        Reward_sum = 0.
        for i in range(self.component_number):
            Reward_sum = Reward_sum + reward(state[i, :], action_index[i], self.state_number)

        belief_state = belief_state.reshape(1, self.component_number * self.state_number)

        return belief_state, Reward_sum, new_hidden_state

class Actor(keras.Model):
    def __init__(self, state_size, action_size, batchnorm=True, hidden=[256, 256]):
        super(Actor, self).__init__()
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        self.fc1 = layers.Dense(hidden[0], input_shape=[None, state_size])
        self.fc2 = layers.Dense(hidden[1])
        self.fc3 = layers.Dense(action_size)
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.batchnorm = batchnorm

    def call(self, x):
        if self.batchnorm:
            x = tf.nn.relu(self.bn1(self.fc1(x)))
            x = tf.nn.relu(self.bn2(self.fc2(x)))
            x = self.fc3(x)
        else:
            x = tf.nn.relu(self.fc1(x))
            x = tf.nn.relu(self.fc2(x))
            x = self.fc3(x)
        return x

class Critic(keras.Model):
    def __init__(self, state_size, action_size, batchnorm=True, hidden=[256, 256]):
        super(Critic, self).__init__()
        self.fc1_state = layers.Dense(hidden[0], input_shape=[None, state_size])
        self.fc1_action = layers.Dense(hidden[0], input_shape=[None, action_size])
        self.fc2 = layers.Dense(hidden[1])
        self.fc3 = layers.Dense(1)
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.batchnorm = batchnorm

    def call(self, state, action):
        if self.batchnorm:
            x_state = tf.nn.relu(self.bn1(self.fc1_state(state)))
            x_action = tf.nn.relu(self.bn1(self.fc1_action(action)))
        else:
            x_state = tf.nn.relu(self.fc1_state(state))
            x_action = tf.nn.relu(self.fc1_action(action))
        x = tf.nn.relu(self.fc2(tf.concat([x_state, x_action], axis=-1)))

        return self.fc3(x)

class ReplayBuffer:
    """Replay Buffer to store transitions."""
    def __init__(self, size=10000, component_number = 16, state_number = 10):
        self.size = size
        self.component_number = component_number
        self.state_number = state_number
        self.input_shape = self.component_number * self.state_number
        self.count = 0  # total index of memory written to, always less than self.size
        self.current = 0  # index to write to

        # Pre-allocate memory
        self.actions = np.empty([self.size, self.component_number], dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.state_old = np.empty([self.size, self.input_shape], dtype=np.float32)
        self.state_new = np.empty([self.size, self.input_shape], dtype=np.float32)

    def add_experience(self, action, state_old, reward, state_new):
        """Saves a transition to the replay buffer
        Args:
            action: An integer between 0 and env.action_space.n - 1
            state_old:
            reward:A float determining the reward the agend received for performing an action
            state_new:
        """

        self.actions[self.current, ...] = action
        self.rewards[self.current] = reward
        self.state_old[self.current, ...] = state_old
        self.state_new[self.current, ...] = state_new
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.size

    def get_minibatch(self, batch_size=32):
        """
        Returns a minibatch of self.batch_size = 32 transitions
            Args:
            batch_size: How many samples to return
        Returns:
             A tuple of states, actions, rewards, new_states, and terminals
         """
        # Get a list of valid indices
        indices = []
        for i in range(batch_size):
            index = np.random.randint(0, self.count - 1)
            indices.append(index)

        # Retrieve states from memory
        states = []
        actions = []
        rewards = []
        new_states = []

        for idx in indices:
            states.append(self.state_old[idx, ...])
            actions.append(self.actions[idx, ...])
            rewards.append(self.rewards[idx])
            new_states.append(self.state_new[idx, ...])

        return states, actions, rewards, new_states

    def save(self, folder_name):
        """Save the replay buffer to a folder"""

        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        np.save(folder_name + '/state_old.npy', self.state_old)
        np.save(folder_name + '/actions.npy', self.actions)
        np.save(folder_name + '/rewards.npy', self.rewards)
        np.save(folder_name + '/state_new.npy', self.state_new)

    def load(self, folder_name):
        """Loads the replay buffer from a folder"""
        self.state_old = np.load(folder_name + '/state_old.npy')
        self.actions = np.load(folder_name + '/actions.npy')
        self.rewards = np.load(folder_name + '/rewards.npy')
        self.state_new = np.load(folder_name + '/state_new.npy')


class DDPG_Agent():
    def __init__(self,
                 state_size,
                 action_size,
                 Main_Actor,
                 Target_Actor,
                 Main_Critic,
                 Target_Critic,
                 Replay_buffer,
                 replay_buffer_start_size,
                 gamma = 0.95,
                 batch_size = 32):
        # state vector and action space
        self.state_size = state_size
        self.action_size = action_size

        # Define DDPG network
        self.Main_Actor = Main_Actor
        self.Target_Actor = Target_Actor
        self.Main_Critic = Main_Critic
        self.Target_Critic = Target_Critic

        self.Replay_buffer = Replay_buffer
        self.replay_buffer_start_size = replay_buffer_start_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.critic_optimizer = tf.optimizers.Adam(0.0005)
        self.actor_optimizer = tf.optimizers.Adam(0.002)

    def get_action(self, state, i):
        action = self.Main_Actor(state).numpy()
        action = tf.sigmoid(action).numpy()
        action_index = np.zeros(self.action_size)
        #noise = 0.1 * np.random.standard_normal(self.action_size)
        #action += noise / np.sqrt(i)
        #action = np.clip(action, 0, 1)
        action = action.reshape(-1)

        # 0_____0.25_____0.5_____0.75_____1

        for i in range(len(action)):
            distance_1 = np.absolute(0 - action[i])
            distance_2 = np.absolute(0.25 - action[i])
            distance_3 = np.absolute(0.5 - action[i])
            distance_4 = np.absolute(0.75 - action[i])
            distance_5 = np.absolute(1 - action[i])

            norm = np.array([(1 - distance_1)*20, (1 - distance_2)*20, (1 - distance_3)*20, (1 - distance_4)*20, (1 - distance_5)*20])
            prob = tf.nn.softmax(norm).numpy()
            action_index[i] = np.random.choice(5, p=prob.ravel())

        action_index = action_index.astype(np.int32)

        return action_index

    def update_target_network(self, Main_network, Target_network, Tau = 0.1):
        Main_network_weight = Main_network.get_weights()
        Target_network_weight = Target_network.get_weights()

        for i in range(len(Main_network_weight)):
            Target_network_weight[i] = Tau * Main_network_weight[i] + (1 - Tau) * Target_network_weight[i]
        Target_network.set_weights(Target_network_weight)

    def learn(self):
        """Sample a batch and use it to improve the actor and critic"""
        states, actions, rewards, new_states = self.Replay_buffer.get_minibatch(batch_size=self.batch_size)
        states, actions, rewards, new_states = np.stack(states), np.stack(actions), np.stack(rewards).reshape(-1,1), np.stack(new_states)
        with tf.GradientTape() as tape:
            expected_rewards = rewards + self.gamma * self.Target_Critic(new_states, self.Target_Actor(new_states))
            estimate_rewards = self.Main_Critic(states, actions)
            loss = tf.keras.losses.mean_squared_error(expected_rewards, estimate_rewards)
            loss = tf.reduce_mean(loss)
        Critic_grads = tape.gradient(loss, self.Main_Critic.trainable_variables)
        #Critic_grads = [tf.clip_by_norm(g, 8) for g in Critic_grads]
        self.critic_optimizer.apply_gradients(zip(Critic_grads,self.Main_Critic.trainable_variables))

        with tf.GradientTape() as tape:
            action_loss = - tf.reduce_mean(self.Main_Critic(states, self.Main_Actor(states)))
        Actor_grads = tape.gradient(action_loss, self.Main_Actor.trainable_variables)
        #print(Actor_grads)
        Actor_grads = [tf.clip_by_norm(g, 8) for g in Actor_grads]
        self.actor_optimizer.apply_gradients(zip(Actor_grads, self.Main_Actor.trainable_variables))

        self.update_target_network(self.Main_Actor, self.Target_Actor, 0.001)
        self.update_target_network(self.Main_Critic, self.Target_Critic, 0.0001)

        return float(loss.numpy()), float(action_loss.numpy())

    def save(self,folder_name):
        """
            Args:
            folder_name: Folder in which to save the Agent
        """
        # Create the folder for saving the agent
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)

        # Save Main_DQN and Target_DQN
        self.Main_Actor.save_weights(folder_name + '/Main_Actor_DDPG')
        self.Target_Actor.save_weights(folder_name + '/Target_Actor_DDPG')
        self.Main_Critic.save_weights(folder_name + '/Main_Critic_DDPG')
        self.Target_Critic.save_weights(folder_name + '/Target_Critic_DDPG')

        # Save replay buffer
        self.Replay_buffer.save(folder_name + '/Replay-buffer')

        # Save the training number and current traning number
        with open('data.txt', 'w+') as f:
            f.write(str(self.Replay_buffer.count) + " " + str(self.Replay_buffer.current))

    def load(self, folder_name, load_replay_buffer=True):
        """Load a previously saved Agent from a folder
            Arguments:
                folder_name: Folder from which to load the Agent
            Returns:
                All other saved attributes, e.g., state number
        """
        if not os.path.isdir(folder_name):
            raise ValueError(f'{folder_name} is not a valid directory')

        # Load Main_DQN and Target DQN
        self.Main_Actor = tf.keras.models.load_model(folder_name + '/Main_Actor_DDPG.h5')
        self.Target_Actor = tf.keras.models.load_model(folder_name + '/Target_Actor_DDPG.h5')
        self.Main_Critic = tf.keras.models.load_model(folder_name + '/Main_Critic_DDPG.h5')
        self.Target_Critic = tf.keras.models.load_model(folder_name + '/Target_Critic_DDPG.h5')

        # Load replay buffer
        if load_replay_buffer:
            self.Replay_buffer.load(folder_name + '/Replay-buffer')

        if load_replay_buffer:
            with open('data.txt', 'r+') as f:
                lines = f.readlines()[0].split(" ")

            self.Replay_buffer.count = int(lines[0])
            self.Replay_buffer.current = int(lines[1])

def main():
    Environment = environment()
    initial_states = Environment.reset()
    Main_Actor = Actor(160,16)
    Target_Actor = Actor(160,16)
    Main_Critic = Critic(160, 16)
    Target_Critic = Critic(160, 16)
    Replay_buffer = ReplayBuffer()
    num_episode = 3000
    max_over_step = 100
    batch_size = 256
    replay_buffer_start_size = 1000
    gamma = 0.95
    initial_hidden_state = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    count = 1

    agent = DDPG_Agent(160, 16, Main_Actor, Target_Actor, Main_Critic, Target_Critic, Replay_buffer,
                       replay_buffer_start_size, gamma=gamma, batch_size=batch_size)
    for i in range(num_episode):
        t = 0
        states = initial_states
        hidden_states = initial_hidden_state
        reward_sum = 0
        while t <= max_over_step:
            t += 1

            actions = agent.get_action(states, count)
            New_belief_state, Rewards, hidden_states = Environment.step(states, actions, hidden_states)
            Replay_buffer.add_experience(actions, states, Rewards, New_belief_state)

            states = New_belief_state
            reward_sum = reward_sum + Rewards

            if Replay_buffer.count > replay_buffer_start_size:
                count += 1
                loss, action_loss = agent.learn()

                if t == max_over_step:
                    print("epoch num:", i, " time stemp: ", t, "   loss: ", loss, "   action_loss: ", action_loss, "   Reward: ", reward_sum)
                    print(actions)
    agent.save('DDPGmodel')

if __name__ == '__main__':
    main()
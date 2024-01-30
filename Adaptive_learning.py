"""
this example illustrates how use the deep reinforcement learning method to solve the large state space and large
discrete action space problem. A transportation network is designed based on the real-world city Bao Ding.
The traffic flows is based on the statistical results. The deterioration law is come from inspection database.

The environment is designed by some modules functions which user can define the bridge numbers, corresponding
state transition matrix, observation matrix, action space.

We should emphasize the action defined part. Normally, the maintenance actions in infrastructural management is
discrete. If a component has three available actions, for multiple-component structure, the total action combinations is
3^N (N is number of component). To avoid the curse of dimensionality, the branching architecture DQN is adopted to

copyright- Lai Li (The Hong Kong Polytechnic University, main programmer)
Wang Aijun (Kuaishou company ,over 140 points in the postgraduate entrance mathematical exams, second programmer)
Prof. Dong You (The Hong Kong Polytechnic University, guidance)
"""

import sys
import numpy as np
import os, random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from tensorflow import keras

class Graph(object):
    def __init__(self, nodes, init_graph):
        self.nodes = nodes
        self.graph = self.construct_graph(nodes, init_graph)

    def construct_graph(self, nodes, init_graph):
        '''
        This method makes sure that the graph is symmetrical. In other words, if there's a path from node A to B with a value V, there needs to be a path from node B to node A with a value V.
        '''
        graph = {}
        for node in nodes:
            graph[node] = {}

        graph.update(init_graph)

        for node, edges in graph.items():
            for adjacent_node, value in edges.items():
                if graph[adjacent_node].get(node, False) == False:
                    graph[adjacent_node][node] = value

        return graph

    def get_nodes(self):
        "Returns the nodes of the graph."
        return self.nodes

    def get_outgoing_edges(self, node):
        "Returns the neighbors of a node."
        connections = []
        for out_node in self.nodes:
            if self.graph[node].get(out_node, False) != False:
                connections.append(out_node)
        return connections

    def value(self, node1, node2):
        "Returns the value of an edge between two nodes."
        return self.graph[node1][node2]


def dijkstra_algorithm(graph, start_node):
    unvisited_nodes = list(graph.get_nodes())

    # We'll use this dict to save the cost of visiting each node and update it as we move along the graph
    shortest_path = {}

    # We'll use this dict to save the shortest known path to a node found so far
    previous_nodes = {}

    # We'll use max_value to initialize the "infinity" value of the unvisited nodes
    max_value = sys.maxsize
    for node in unvisited_nodes:
        shortest_path[node] = max_value
    # However, we initialize the starting node's value with 0
    shortest_path[start_node] = 0

    # The algorithm executes until we visit all nodes
    while unvisited_nodes:
        # The code block below finds the node with the lowest score
        current_min_node = None
        for node in unvisited_nodes:  # Iterate over the nodes
            if current_min_node == None:
                current_min_node = node
            elif shortest_path[node] < shortest_path[current_min_node]:
                current_min_node = node

        # The code block below retrieves the current node's neighbors and updates their distances
        neighbors = graph.get_outgoing_edges(current_min_node)
        for neighbor in neighbors:
            tentative_value = shortest_path[current_min_node] + graph.value(current_min_node, neighbor)
            if tentative_value < shortest_path[neighbor]:
                shortest_path[neighbor] = tentative_value
                # We also update the best path to the current node
                previous_nodes[neighbor] = current_min_node

        # After visiting its neighbors, we mark the node as "visited"
        unvisited_nodes.remove(current_min_node)

    return previous_nodes, shortest_path


def print_result(previous_nodes, shortest_path, start_node, target_node):
    path = []
    node = target_node

    while node != start_node:
        path.append(node)
        node = previous_nodes[node]

    # Add the start node manually
    path.append(start_node)

    print("We found the following best path with a value of {}.".format(shortest_path[target_node]))
    print(path)


def state_transition(state_number, prob):
    """ Markov model state transition matrix """
    transition_matrix = np.zeros((1, state_number, state_number))
    transition_matrix[0, state_number - 1, state_number - 1] = 1

    for step, ri in enumerate(prob):
        Pi = ri / (1 + ri)
        Qi = 1 / (1 + ri)
        transition_matrix[0, step, step] = Pi
        transition_matrix[0, step, step + 1] = Qi
    return transition_matrix

def observation(accuracy, state_number):
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
    for step in range(state_number - 2):
        observation_matrix[step + 1, step] = (1 - accuracy) / 2
        observation_matrix[step + 1, step + 1] = accuracy
        observation_matrix[step + 1, step + 2] = (1 - accuracy) / 2

    return observation_matrix

def action_matrix(action_index, state_number):
    """
    based on the utility value, the
    Args:
        action_index: 0 means Inspection, 2 means medium repair, 3 means replacement,
        state_number: the state number of a component
    Returns:
        repair matrix when agent execute the maintenance action with utility value
    """
    repair_matrix = np.zeros((state_number, state_number))
    if action_index == 0:
        repair_matrix = np.eye(state_number, dtype=int)
    elif action_index == 1:
        repair_matrix = np.array([[1, 0, 0, 0, 0],
                                  [1, 0, 0, 0, 0],
                                  [0.05, 0.95, 0, 0, 0],
                                  [0.05, 0.1, 0.85, 0, 0],
                                  [0, 0.05, 0.15, 0.8, 0]])
    elif action_index == 2:
        repair_matrix[:, 0] = 1

    return repair_matrix

def reward(component_state, actions, state_number, Total_CO2, mobility):
    """
    Args:
        component_state: a vector represent all strutural state (component_number * state_number)
        actions: 0-repair 0.5-inspection 1-replace (a vector, size = component_number)
        state_number: 0 means intact, 4 means failure
        Total_CO2: CO2 emission from vehicles, unit ton
        mobility: the average grade of service in the current transportation network
        bridge_span : each bridge span, which will determine the maintenance fee for bridge

    Returns:
        normalized reward 0-1
    """
    # Reward table is the immediate reward based on the states (column), actions (row) and bridge span
    Reward_table = np.zeros((3, state_number))

    # the punishment for the state deterioration
    state_deterioration_punishment = 1.02
    # unit CNY (K)
    inspection = -1
    repair = -16
    replace = -20
    punishment = -60
    cost = np.array([inspection, repair, replace])
    for i in range(len(cost)):
        for j in range(state_number):
            Reward_table[i, j] = cost[i] * state_deterioration_punishment ** j
    Reward_table[:, state_number - 1] = punishment

    Reward_maintenance = np.zeros((len(actions)))

    for i in range(len(actions)):
        for j in range(state_number):
            if actions[i] == 0:
                Reward_maintenance[i] += Reward_table[0, j] * component_state[i, j]
    # * bridge_span[i]
            elif actions[i] == 1:
                Reward_maintenance[i] += Reward_table[1, j] * component_state[i, j]
            elif actions[i] == 2:
                Reward_maintenance[i] += Reward_table[2, j] * component_state[i, j]
    Reward = np.sum(Reward_maintenance)

    # normalized for bridge maintenance fee
    normalized_fee = - Reward/ (-20 * 7)

    normalized_Reward_maintenance = - Reward_maintenance/(-20)

    # normalized for CO2 emission
    # the baseline for CO2 emission is 3.1e3 ton/day
    # the maximum CO2 emission is 4.62 ton/day
    CO2_normalization = - (Total_CO2 - 3.1e3 * 365) / (6.2e3 * 365 - 3.1e3 * 365)

    # note the negative
    normalized_Reward = 0.7 * normalized_fee + 0.15 * CO2_normalization + 0.15 * (mobility - 1)

    return normalized_Reward, normalized_Reward_maintenance

def reward_branch(component_state, actions, state_number):
    """
    Args:
        component_state: a vector represent all strutural state (component_number * state_number)
        actions: 0-repair 0.5-inspection 1-replace (a vector, size = component_number)
        state_number: 0 means intact, 4 means failure
        bridge_span : each bridge span, which will determine the maintenance fee for bridge

    Returns:
        normalized reward 0-1
    """
    # Reward table is the immediate reward based on the states (column), actions (row) and bridge span
    Reward_table = np.zeros((3, state_number))

    # the punishment for the state deterioration
    state_deterioration_punishment = 1.02
    # unit CNY (K)
    inspection = -1
    repair = -16
    replace = -20
    punishment = -60
    cost = np.array([inspection, repair, replace])
    for i in range(len(cost)):
        for j in range(state_number):
            Reward_table[i, j] = cost[i] * state_deterioration_punishment ** j
    Reward_table[:, state_number - 1] = punishment

    Reward_maintenance = np.zeros((len(actions)))

    for i in range(len(actions)):
        for j in range(state_number):
            if actions[i] == 0:
                Reward_maintenance[i] += Reward_table[0, j] * component_state[i, j]
    # * bridge_span[i]
            elif actions[i] == 1:
                Reward_maintenance[i] += Reward_table[1, j] * component_state[i, j]
            elif actions[i] == 2:
                Reward_maintenance[i] += Reward_table[2, j] * component_state[i, j]

    normalized_Reward_maintenance = - Reward_maintenance/(-20)

    return  normalized_Reward_maintenance

class environment():
    """this part define the bridge degradation process"""

    def __init__(self):
        """fundamental parameters of structure or bridges"""
        self.component_number = 14
        self.span = np.array([6, 12, 41.4, 36, 96, 245, 66.4, 51.5, 52.5, 106.8, 23, 40, 484.1, 55])
        self.state_number = 5
        self.accuracy = 0.9
        # concrete bridge
        self.prob1 = np.array([0.922/0.078, 0.931/0.069, 0.94/0.06, 19])
        # prestress concrete bridge
        self.prob2 = np.array([0.906/0.094, 0.957/0.043, 0.95/0.05, 0.94/0.06])

        self.transition_matrix = np.zeros((self.component_number, self.state_number, self.state_number))

        for i in range(self.component_number):
            if i != 6:
                self.transition_matrix[i, :, :] = state_transition(self.state_number,
                                                                   self.prob1.reshape(self.state_number - 1, 1))
            elif i == 6:
                self.transition_matrix[i, :, :] = state_transition(self.state_number,
                                                                   self.prob2.reshape(self.state_number - 1, 1))

        self.Observation = observation(self.accuracy, self.state_number)

    def reset(self):
        self.state = np.zeros((self.component_number, self.state_number))
        self.state[[0, 1, 2, 3, 5, 6, 8, 10, 11, 13], 0] = 1
        self.state[[4, 7, 9, 12], 1] = 1
        self.state = self.state.reshape(-1, self.component_number * self.state_number)

        return self.state

    def step(self, states, action_index, hidden_state):
        state = np.reshape(states, [self.component_number, -1])
        new_hidden_state = np.zeros((1, self.component_number), dtype=int)
        # this is Bayesian updating in POMDP
        state_update = np.zeros((self.component_number, self.state_number))
        for i in range(self.component_number):
            repair_matrix = action_matrix(action_index[i], self.state_number)
            if action_index[i] == 0:
                transition_matrix = self.transition_matrix[i, :, :]
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
            belief_state[i, :] = belief_state[i, :] / np.sum(belief_state[i, :])
        # immediate reward part r

        # Bridge condition state, which will determine whether the large vehicle will be restricted
        expect_con = np.zeros((self.component_number))
        for i in range(self.component_number):
            for j in range(self.state_number):
                expect_con[i] +=  (j + 1) * belief_state[i, j]

        # establish a transportation network for small and medium truck
        nodes = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]

        init_graph = {}
        for node in nodes:
            init_graph[node] = {}

        if expect_con[0] < 4 and expect_con[1] < 4 and action_index[0] != 2 and action_index[1] != 2:
            init_graph["A"]["B"] = 23
        if expect_con[2] < 4 and action_index[2] != 2:
            init_graph["B"]["D"] = 24.8
        if expect_con[3] < 4 and action_index[3] != 2:
            init_graph["B"]["E"] = 22.2
        if expect_con[4] < 4 and expect_con[5] < 4 and action_index[4] != 2 and action_index[5] != 2:
            init_graph["A"]["F"] = 28.7
        if expect_con[6] < 4 and expect_con[7] < 4 and action_index[6] != 2 and action_index[7] != 2:
            init_graph["E"]["H"] = 25.1
        if expect_con[8] < 4 and action_index[8] != 2:
            init_graph["E"]["F"] = 30.1
        if expect_con[9] < 4 and action_index[9] != 2:
            init_graph["F"]["G"] = 17.2
        if expect_con[10] < 4 and action_index[10] != 2:
            init_graph["G"]["H"] = 19.7
        if expect_con[11] < 4 and action_index[11] != 2:
            init_graph["H"]["I"] = 22.7
        if expect_con[12] < 4 and expect_con[13] < 4 and action_index[12] != 2 and action_index[13] != 2:
            init_graph["G"]["I"] = 21.4
        init_graph["C"]["D"] = 11.7
        init_graph["C"]["E"] = 11.7

        graph = Graph(nodes, init_graph)
        A_car_route, D0 = dijkstra_algorithm(graph=graph, start_node="A")
        B_car_route, D1 = dijkstra_algorithm(graph=graph, start_node="B")
        C_car_route, D2 = dijkstra_algorithm(graph=graph, start_node="C")
        D_car_route, D3 = dijkstra_algorithm(graph=graph, start_node="D")
        E_car_route, D4 = dijkstra_algorithm(graph=graph, start_node="E")
        F_car_route, D5 = dijkstra_algorithm(graph=graph, start_node="F")
        G_car_route, D6 = dijkstra_algorithm(graph=graph, start_node="G")
        H_car_route, D7 = dijkstra_algorithm(graph=graph, start_node="H")
        I_car_route, D8 = dijkstra_algorithm(graph=graph, start_node="I")
        del graph
        # print(A_car_route)
        # print_result(A_car_route, D0, start_node="A", target_node="C")

        Car_trip_matrix = np.array([[D0['A'], D0['B'], D0['C'], D0['D'], D0['E'], D0['F'], D0['G'], D0['H'], D0['I']],
                                    [D1['A'], D1['B'], D1['C'], D1['D'], D1['E'], D1['F'], D1['G'], D1['H'], D1['I']],
                                    [D2['A'], D2['B'], D2['C'], D2['D'], D2['E'], D2['F'], D2['G'], D2['H'], D2['I']],
                                    [D3['A'], D3['B'], D3['C'], D3['D'], D3['E'], D3['F'], D3['G'], D3['H'], D3['I']],
                                    [D4['A'], D4['B'], D4['C'], D4['D'], D4['E'], D4['F'], D4['G'], D4['H'], D4['I']],
                                    [D5['A'], D5['B'], D5['C'], D5['D'], D5['E'], D5['F'], D5['G'], D5['H'], D5['I']],
                                    [D6['A'], D6['B'], D6['C'], D6['D'], D6['E'], D6['F'], D6['G'], D6['H'], D6['I']],
                                    [D7['A'], D7['B'], D7['C'], D7['D'], D7['E'], D7['F'], D7['G'], D7['H'], D7['I']],
                                    [D8['A'], D8['B'], D8['C'], D8['D'], D8['E'], D8['F'], D8['G'], D8['H'], D8['I']]])
        # print(Car_trip_matrix)
        # establish a transportation network for large truck
        init_graph = {}
        for node in nodes:
            init_graph[node] = {}

        if expect_con[0] <= 3 and expect_con[1] <= 3 and action_index[0] != 2 and action_index[1] != 2:
            init_graph["A"]["B"] = 23
        if expect_con[2] <= 3 and action_index[2] != 2:
            init_graph["B"]["D"] = 24.8
        if expect_con[3] <= 3 and action_index[3] != 2:
            init_graph["B"]["E"] = 22.2
        if expect_con[4] <= 3 and expect_con[5] <= 3 and action_index[4] != 2 and action_index[5] != 2:
            init_graph["A"]["F"] = 28.7
        if expect_con[6] <= 3 and expect_con[7] <= 3 and action_index[6] != 2 and action_index[7] != 2:
            init_graph["E"]["H"] = 25.1
        if expect_con[8] <= 3 and action_index[8] != 2:
            init_graph["E"]["F"] = 30.1
        if expect_con[9] <= 3 and action_index[9] != 2:
            init_graph["F"]["G"] = 17.2
        if expect_con[10] <= 3 and action_index[10] != 2:
            init_graph["G"]["H"] = 19.7
        if expect_con[11] <= 3 and action_index[11] != 2:
            init_graph["H"]["I"] = 22.7
        if expect_con[12] <= 3 and expect_con[13] <= 3 and action_index[12] != 2 and action_index[13] != 2:
            init_graph["G"]["I"] = 21.4
        init_graph["C"]["D"] = 11.7
        init_graph["C"]["E"] = 11.7

        graph = Graph(nodes, init_graph)
        A_truck_route, D0 = dijkstra_algorithm(graph=graph, start_node="A")
        B_truck_route, D1 = dijkstra_algorithm(graph=graph, start_node="B")
        C_truck_route, D2 = dijkstra_algorithm(graph=graph, start_node="C")
        D_truck_route, D3 = dijkstra_algorithm(graph=graph, start_node="D")
        E_truck_route, D4 = dijkstra_algorithm(graph=graph, start_node="E")
        F_truck_route, D5 = dijkstra_algorithm(graph=graph, start_node="F")
        G_truck_route, D6 = dijkstra_algorithm(graph=graph, start_node="G")
        H_truck_route, D7 = dijkstra_algorithm(graph=graph, start_node="H")
        I_truck_route, D8 = dijkstra_algorithm(graph=graph, start_node="I")

        del graph
        Truck_trip_matrix = np.array([[D0['A'], D0['B'], D0['C'], D0['D'], D0['E'], D0['F'], D0['G'], D0['H'], D0['I']],
                                      [D1['A'], D1['B'], D1['C'], D1['D'], D1['E'], D1['F'], D1['G'], D1['H'], D1['I']],
                                      [D2['A'], D2['B'], D2['C'], D2['D'], D2['E'], D2['F'], D2['G'], D2['H'], D2['I']],
                                      [D3['A'], D3['B'], D3['C'], D3['D'], D3['E'], D3['F'], D3['G'], D3['H'], D3['I']],
                                      [D4['A'], D4['B'], D4['C'], D4['D'], D4['E'], D4['F'], D4['G'], D4['H'], D4['I']],
                                      [D5['A'], D5['B'], D5['C'], D5['D'], D5['E'], D5['F'], D5['G'], D5['H'], D5['I']],
                                      [D6['A'], D6['B'], D6['C'], D6['D'], D6['E'], D6['F'], D6['G'], D6['H'], D6['I']],
                                      [D7['A'], D7['B'], D7['C'], D7['D'], D7['E'], D7['F'], D7['G'], D7['H'], D7['I']],
                                      [D8['A'], D8['B'], D8['C'], D8['D'], D8['E'], D8['F'], D8['G'], D8['H'], D8['I']]])
        # judge the effective transportation network
        # np.allclose(Truck_trip_matrix, Truck_trip_matrix.transpose(), rtol=1e-5, atol=1e-8, equal_nan=False)

        # judge the transportation network whether works
        if (Truck_trip_matrix > 1e4).any() or (Car_trip_matrix > 1e4).any():
            Reward_sum = -0.7
            normalized_Reward_maintenance = reward_branch(belief_state, action_index, self.state_number)
        else:
            """
                    This function defines the traffic flow from starting place to destination
                        Args:
                        start_point: (0), (2), (3), (4), (5), (7), (8)
                        end_point: (0), (2), (3), (4), (5), (7), (8)
                    Returns:
                    """
            traffic_flow_matrix = np.array([[0, 0, 6133, 18503, 15235, 2102, 0, 6214, 11365],
                                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                            [6133, 0, 0, 1002, 612, 1342, 0, 1757, 2361],
                                            [18503, 0, 1002, 0, 1034, 3014, 0, 1439, 12219],
                                            [15235, 0, 612, 1034, 0, 3985, 0, 2971, 3649],
                                            [2102, 0, 1342, 3014, 3985, 0, 0, 12006, 8640],
                                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                            [6214, 0, 1757, 1439, 2971, 12006, 0, 0, 3806],
                                            [11365, 0, 2361, 12219, 3649, 8640, 0, 3806, 0]])

            # the ratio of small car and large size vehicle in total traffic flow
            Car_ratio = 0.46
            Truck_ratio = 0.54

            Car_flow_matrix = Car_ratio * traffic_flow_matrix
            Truck_flow_matrix = Truck_ratio * traffic_flow_matrix

            """
            Based on the fuel consuming and different combustion ratio, the C02 = fuel consuming * combustion ratio
            Args:
                car_volume: total traffic flow (/day)
                distance: between starting place and destination (km)
                0 : small truck;
                1 : Medium duty truck;
                2 : Heavy duty vehicle;
                3 : extreme truck;
                4 : Delivery van;
                5 : medium bus;
                6 : Long distance bus;
                7 : means Motorcycle;
                8 : Tractor;
            Returns: the CO2 emission (g)

            """
            CO2 = np.array([276, 450, 867.9, 1236.1, 1315, 252, 644, 80.5, 157.8])
            # define the coefficient of different type in traffic flow
            Car_coefficient = np.array([0.1197, 0.0813, 0, 0, 0, 0.4846, 0, 0.1299, 0.0359])
            Truck_coefficient = np.array([0, 0, 0.0624, 0.1346, 0.0483, 0, 0.0539, 0, 0])
            # calculate the CO2 emission whole year (g)
            Car_CO2 = np.sum(Car_trip_matrix * Car_flow_matrix) * np.sum(CO2 * Car_coefficient)
            Truck_CO2 = np.sum(Truck_trip_matrix * Truck_flow_matrix) * np.sum(CO2 * Truck_coefficient)
            # Total of CO2 emission
            Total_CO2 = 365 * (Car_CO2 + Truck_CO2) / 1e6
            # print(Total_CO2)

            # calculate the traffic volume in each traffic lane
            traffic_lane_matrix = np.zeros((len(nodes), len(nodes)))
            Car_percentage = np.sum(Car_coefficient)
            Truck_percentage = np.sum(Truck_coefficient)

            # start from point A
            destination = ["C", "D", "E", "F", "H", "I"]
            Car_flow_matrix_A = np.nonzero(Car_flow_matrix[0, :])
            Car_flow_A = np.reshape(Car_flow_matrix[0, Car_flow_matrix_A], -1) * Car_percentage
            Truck_flow_matrix_A = np.nonzero(Truck_flow_matrix[0, :])
            Truck_flow_A = np.reshape(Truck_flow_matrix[0, Truck_flow_matrix_A], -1) * Truck_percentage

            for i in range(len(destination)):
                path_car = []
                node_car = destination[i]
                car_number = Car_flow_A[i]

                path_truck = []
                node_truck = destination[i]
                truck_number = Truck_flow_A[i]

                while node_car != 'A':
                    path_car.append(node_car)
                    node_car = A_car_route[node_car]

                path_car.append('A')
                path_car = np.flipud(path_car)
                for j in range(len(path_car) - 1):
                    row = ord(path_car[j]) - 65
                    column = ord(path_car[j + 1]) - 65
                    traffic_lane_matrix[row, column] = car_number + traffic_lane_matrix[row, column]

                while node_truck != 'A':
                    path_truck.append(node_truck)
                    node_truck = A_truck_route[node_truck]

                path_truck.append('A')
                path_truck = np.flipud(path_truck)
                for j in range(len(path_truck) - 1):
                    row = ord(path_truck[j]) - 65
                    column = ord(path_truck[j + 1]) - 65
                    traffic_lane_matrix[row, column] = truck_number + traffic_lane_matrix[row, column]

            # start from Point C
            destination = ["D", "E", "F", "H", "I"]
            Car_flow_matrix_C = np.nonzero(Car_flow_matrix[2, :])
            Car_flow_C = np.reshape(Car_flow_matrix[2, Car_flow_matrix_C], -1) * Car_percentage
            Car_flow_C = Car_flow_C[-5:]
            Truck_flow_matrix_C = np.nonzero(Truck_flow_matrix[2, :])
            Truck_flow_C = np.reshape(Truck_flow_matrix[2, Truck_flow_matrix_C], -1) * Truck_percentage
            Truck_flow_C = Truck_flow_C[-5:]
            for i in range(len(destination)):
                path_car = []
                node_car = destination[i]
                car_number = Car_flow_C[i]

                path_truck = []
                node_truck = destination[i]
                truck_number = Truck_flow_C[i]

                while node_car != 'C':
                    path_car.append(node_car)
                    node_car = C_car_route[node_car]

                path_car.append('C')
                path_car = np.flipud(path_car)
                for j in range(len(path_car) - 1):
                    row = ord(path_car[j]) - 65
                    column = ord(path_car[j + 1]) - 65
                    traffic_lane_matrix[row, column] = car_number + traffic_lane_matrix[row, column]

                while node_truck != 'C':
                    path_truck.append(node_truck)
                    node_truck = C_truck_route[node_truck]

                path_truck.append('C')
                path_truck = np.flipud(path_truck)
                for j in range(len(path_truck) - 1):
                    row = ord(path_truck[j]) - 65
                    column = ord(path_truck[j + 1]) - 65
                    traffic_lane_matrix[row, column] = truck_number + traffic_lane_matrix[row, column]

            # start from Point D
            destination = ["E", "F", "H", "I"]
            Car_flow_matrix_D = np.nonzero(Car_flow_matrix[3, :])
            Car_flow_D = np.reshape(Car_flow_matrix[3, Car_flow_matrix_D], -1) * Car_percentage
            Car_flow_D = Car_flow_D[-4:]
            Truck_flow_matrix_D = np.nonzero(Truck_flow_matrix[3, :])
            Truck_flow_D = np.reshape(Truck_flow_matrix[3, Truck_flow_matrix_D], -1) * Truck_percentage
            Truck_flow_D = Truck_flow_D[-4:]

            for i in range(len(destination)):
                path_car = []
                node_car = destination[i]
                car_number = Car_flow_D[i]

                path_truck = []
                node_truck = destination[i]
                truck_number = Truck_flow_D[i]

                while node_car != 'D':
                    path_car.append(node_car)
                    node_car = D_car_route[node_car]

                path_car.append('D')
                path_car = np.flipud(path_car)
                for j in range(len(path_car) - 1):
                    row = ord(path_car[j]) - 65
                    column = ord(path_car[j + 1]) - 65
                    traffic_lane_matrix[row, column] = car_number + traffic_lane_matrix[row, column]

                while node_truck != 'D':
                    path_truck.append(node_truck)
                    node_truck = D_truck_route[node_truck]

                path_truck.append('D')
                path_truck = np.flipud(path_truck)
                for j in range(len(path_truck) - 1):
                    row = ord(path_truck[j]) - 65
                    column = ord(path_truck[j + 1]) - 65
                    traffic_lane_matrix[row, column] = truck_number + traffic_lane_matrix[row, column]

            # start from Point E
            destination = ["F", "H", "I"]
            Car_flow_matrix_E = np.nonzero(Car_flow_matrix[4, :])
            Car_flow_E = np.reshape(Car_flow_matrix[4, Car_flow_matrix_E], -1) * Car_percentage
            Car_flow_E = Car_flow_E[-3:]
            Truck_flow_matrix_E = np.nonzero(Truck_flow_matrix[4, :])
            Truck_flow_E = np.reshape(Truck_flow_matrix[4, Truck_flow_matrix_E], -1) * Truck_percentage
            Truck_flow_E = Truck_flow_E[-3:]

            for i in range(len(destination)):
                path_car = []
                node_car = destination[i]
                car_number = Car_flow_E[i]

                path_truck = []
                node_truck = destination[i]
                truck_number = Truck_flow_E[i]

                while node_car != 'E':
                    path_car.append(node_car)
                    node_car = E_car_route[node_car]

                path_car.append('E')
                path_car = np.flipud(path_car)
                for j in range(len(path_car) - 1):
                    row = ord(path_car[j]) - 65
                    column = ord(path_car[j + 1]) - 65
                    traffic_lane_matrix[row, column] = car_number + traffic_lane_matrix[row, column]

                while node_truck != 'E':
                    path_truck.append(node_truck)
                    node_truck = E_truck_route[node_truck]

                path_truck.append('E')
                path_truck = np.flipud(path_truck)
                for j in range(len(path_truck) - 1):
                    row = ord(path_truck[j]) - 65
                    column = ord(path_truck[j + 1]) - 65
                    traffic_lane_matrix[row, column] = truck_number + traffic_lane_matrix[row, column]

            # start from Point F
            destination = ["H", "I"]
            Car_flow_matrix_F = np.nonzero(Car_flow_matrix[5, :])
            Car_flow_F = np.reshape(Car_flow_matrix[5, Car_flow_matrix_F], -1) * Car_percentage
            Car_flow_F = Car_flow_F[-2:]
            Truck_flow_matrix_F = np.nonzero(Truck_flow_matrix[5, :])
            Truck_flow_F = np.reshape(Truck_flow_matrix[5, Truck_flow_matrix_F], -1) * Truck_percentage
            Truck_flow_F = Truck_flow_F[-2:]

            for i in range(len(destination)):
                path_car = []
                node_car = destination[i]
                car_number = Car_flow_F[i]

                path_truck = []
                node_truck = destination[i]
                truck_number = Truck_flow_F[i]

                while node_car != 'F':
                    path_car.append(node_car)
                    node_car = F_car_route[node_car]

                path_car.append('F')
                path_car = np.flipud(path_car)
                for j in range(len(path_car) - 1):
                    row = ord(path_car[j]) - 65
                    column = ord(path_car[j + 1]) - 65
                    traffic_lane_matrix[row, column] = car_number + traffic_lane_matrix[row, column]

                while node_truck != 'F':
                    path_truck.append(node_truck)
                    node_truck = F_truck_route[node_truck]

                path_truck.append('F')
                path_truck = np.flipud(path_truck)
                for j in range(len(path_truck) - 1):
                    row = ord(path_truck[j]) - 65
                    column = ord(path_truck[j + 1]) - 65
                    traffic_lane_matrix[row, column] = truck_number + traffic_lane_matrix[row, column]

            # start from Point H
            destination = ["I"]
            Car_flow_matrix_H = np.nonzero(Car_flow_matrix[7, :])
            Car_flow_H = np.reshape(Car_flow_matrix[7, Car_flow_matrix_H], -1) * Car_percentage
            Car_flow_H = Car_flow_H[-1:]
            Truck_flow_matrix_H = np.nonzero(Truck_flow_matrix[7, :])
            Truck_flow_H = np.reshape(Truck_flow_matrix[7, Truck_flow_matrix_H], -1) * Truck_percentage
            Truck_flow_H = Truck_flow_H[-1:]

            for i in range(len(destination)):
                path_car = []
                node_car = destination[i]
                car_number = Car_flow_H[i]

                path_truck = []
                node_truck = destination[i]
                truck_number = Truck_flow_H[i]

                while node_car != 'H':
                    path_car.append(node_car)
                    node_car = H_car_route[node_car]

                path_car.append('H')
                path_car = np.flipud(path_car)
                for j in range(len(path_car) - 1):
                    row = ord(path_car[j]) - 65
                    column = ord(path_car[j + 1]) - 65
                    traffic_lane_matrix[row, column] = car_number + traffic_lane_matrix[row, column]

                while node_truck != 'H':
                    path_truck.append(node_truck)
                    node_truck = H_truck_route[node_truck]

                path_truck.append('H')
                path_truck = np.flipud(path_truck)
                for j in range(len(path_truck) - 1):
                    row = ord(path_truck[j]) - 65
                    column = ord(path_truck[j + 1]) - 65
                    # print(row, column)
                    traffic_lane_matrix[row, column] = truck_number + traffic_lane_matrix[row, column]

            """estimate the grade of service in this transportation network"""
            # first-class highway 3 level -> <24000; 4 level -> 24000~31000; 5 level -> 31000~35000; 6 level -> >35000
            # second-class highway 4 level -> <22000; 5 level -> 22 i8000~ 37000; 6 level -> >37000
            # the estimated equation is C_d = MSF * f_HV * f_p * f_f
            # MSF is maximum traffic for single line, which is 1250 for 80 km/h 3 level, 1600 for 80 km/h 4 level in first-class
            # f_HV is the traffic composition coefficient, f_HV = 1/(1+sum p_i * (E_i - 1)); E_i = 2, 3, 5 from middle to extreme
            # f_p is driver coefficient = 0.98; f_f is the intersection coefficient = 0.9

            traffic_lane = np.triu(traffic_lane_matrix + traffic_lane_matrix.T)
            traffic_volume = traffic_lane.ravel()[np.flatnonzero(traffic_lane)]
            service_grade = np.zeros((len(traffic_volume)))

            for i in range(len(traffic_volume)):
                if traffic_volume[i] < 24000 and i < 2:
                    service_grade[i] = 1
                elif traffic_volume[i] >= 24000 and traffic_volume[i] < 31000 and i < 2:
                    service_grade[i] = 0.66
                elif traffic_volume[i] >= 31000 and traffic_volume[i] < 35000 and i < 2:
                    service_grade[i] = 0.33
                elif traffic_volume[i] >= 35000 and i < 2:
                    service_grade[i] = 0
                elif traffic_volume[i] < 22000 and i >= 2:
                    service_grade[i] = 1
                elif traffic_volume[i] >= 22000 and traffic_volume[i] < 37000 and i >= 2:
                    service_grade[i] = 0.5
                elif traffic_volume[i] >= 37000 and i >= 2:
                    service_grade[i] = 0

            mobility = service_grade.mean()
            #print( - (Total_CO2 - 3.1e3 * 365) / (4.62e3 * 365 - 3.1e3 * 365))
            #print(mobility)
            Reward_sum, normalized_Reward_maintenance  = reward(belief_state, action_index, self.state_number, Total_CO2, mobility)

        belief_state = belief_state.reshape(1, self.component_number * self.state_number)

        return belief_state, Reward_sum, new_hidden_state, normalized_Reward_maintenance

class BDQN(keras.Model):
    def __init__(self, state_size=5, num_action_branches=14, commom_hidden=[256, 128], branch_hidden = [64, 32, 3]):
        super(BDQN, self).__init__()
        self.fc1 = layers.Dense(commom_hidden[0], name="dense_1", input_shape=[None, state_size])
        self.fc2 = layers.Dense(commom_hidden[1], name="dense_2")
        self.fc3_list = [layers.Dense(branch_hidden[0], name="branch_layer1_"+str(i)) for i in range(num_action_branches)]
        self.fc4_list = [layers.Dense(branch_hidden[1], name="branch_layer2_"+str(i)) for i in range(num_action_branches)]
        self.fcout_list = [layers.Dense(branch_hidden[2], name="branch_layer3_" + str(i)) for i in
                         range(num_action_branches)]
        self.fc5 = layers.Dense(64)
        self.fc6 = layers.Dense(1)
        # using layers.Lambda to define a new layer which not includes in keras
        self.lambda_layer = layers.Lambda(lambda x: x - tf.reduce_mean(x))
        self.combine = layers.Add()
        self.num_action_branches = num_action_branches

    def call(self, x):
        # common hidden layer 14 * 5 = 70 -> 256
        x = tf.nn.relu(self.fc1(x))
        # x = tf.nn.dropout(x, 0.2)
        # Shared representation 256 -> 128
        x = tf.nn.relu(self.fc2(x))
        # x = tf.nn.dropout(x, 0.2)

        # current state value V(s) 128 -> 64 -> 1
        x_0 = tf.nn.relu(self.fc5(x))
        # x_0 = tf.nn.dropout(x_0, 0.2) don't use relU
        value = self.fc6(x_0)  # 128 -> 1

        # advantage for each action dimension
        total_action_scores = []
        for index in range(self.num_action_branches):
            # action branching 128 -> 64
            action_hidden = tf.nn.relu(self.fc3_list[index](x))
            # action branching 64 -> 32
            action_out = tf.nn.relu(self.fc4_list[index](action_hidden))
            # action branching 32 -> 3 Don't use relU
            action_out = self.fcout_list[index](action_out)
            # reduce LocalMean
            action_scores = self.lambda_layer(action_out)

            total_action_scores.append(action_scores)
        total_Q_value = [self.combine([value, action_score]) for action_score in total_action_scores]
        total_Q_value = tf.stack(total_Q_value, axis=1)

        return total_Q_value

"""
SumTree : save TD ERROR
"""
class SumTree:

    def __init__(self, capacity):

        self.capacity = capacity
        # the first capacity-1 positions are not leaves
        self.vals = [0 for _ in range(2 * capacity - 1)]  # think about why if you are not familiar with this

    def retrive(self, num):
        '''
        This function find the first index whose cumsum is no smaller than num
        '''
        ind = 0  # search from root
        while ind < self.capacity - 1:  # not a leaf
            left = 2 * ind + 1
            right = left + 1
            if num > self.vals[left]:  # the sum of the whole left tree is not large enouth
                num -= self.vals[left]  # think about why?
                ind = right
            else:  # search in the left tree
                ind = left
        return ind - self.capacity + 1

    def update(self, delta, ind):
        '''
        Change the value at ind by delta, and update the tree
        Notice that this ind should be the index in real memory part, instead of the ind in self.vals
        '''
        ind += self.capacity - 1
        while True:
            self.vals[ind] += delta
            if ind == 0:
                break
            ind -= 1
            ind //= 2

class ReplayBuffer:
    """Replay Buffer to store transitions."""
    def __init__(self, size=5000, component_number=14, state_number=5):
        self.size = size
        self.component_number = component_number
        self.state_number = state_number
        self.input_shape = self.component_number * self.state_number
        self.count = 0  # total index of memory written to, always less than self.size
        self.current = 0  # index to write to

        # Pre-allocate memory
        self.actions = np.empty([self.size, self.component_number], dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.rewards_maintenance = np.empty([self.size, self.component_number], dtype=np.float32)
        self.state_old = np.empty([self.size, self.input_shape], dtype=np.float32)
        self.state_new = np.empty([self.size, self.input_shape], dtype=np.float32)

        self.weights = SumTree(self.size)
        self.alpha = 1  # Pi**α
        self.td_error = 1  # td error init value
        self.ind_max = 0
        self.epsilon = 0.01
    def add_experience(self, action, state_old, reward, state_new, reward_maintenance):
        """Saves a transition to the replay buffer
        Args:
            action: An integer between 0 and env.action_space.n - 1
            state_old:
            reward:A float determining the reward the agend received for performing an action
            state_new:
        """

        self.actions[self.current, ...] = action
        self.rewards[self.current] = reward
        self.rewards_maintenance[self.current, ...] = reward_maintenance
        self.state_old[self.current, ...] = state_old
        self.state_new[self.current, ...] = state_new
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.size

        index = self.ind_max % self.size
        delta = self.td_error + self.epsilon - self.weights.vals[index + self.size - 1]
        self.weights.update(delta, index)  # 更新td error
        self.ind_max += 1
    def get_minibatch(self, batch_size=32):
        """
        Returns a minibatch of self.batch_size = 32 transitions
            Args:
            batch_size: How many samples to return
        Returns:
             A tuple of states, actions, rewards, new_states, and terminals
         """

        #
        indices = [self.weights.retrive(self.weights.vals[0] * random.random()) for _ in range(batch_size)]
        # Pi**α/sum(pi**α)
        probs = [self.weights.vals[ind + self.size - 1] / self.weights.vals[0] for ind in indices]

        #  Get a list of valid indices
        # indices = []
        # for i in range(batch_size):
        #     index = np.random.randint(0, self.count - 1)
        #     indices.append(index)

        # Retrieve states from memory
        states = []
        actions = []
        rewards = []
        new_states = []
        reward_maintenance = []

        for idx in indices:
            states.append(self.state_old[idx, ...])
            actions.append(self.actions[idx, ...])
            rewards.append(self.rewards[idx])
            new_states.append(self.state_new[idx, ...])
            reward_maintenance.append(self.rewards_maintenance[idx, ...])

        return states, actions, rewards, new_states, reward_maintenance, indices, probs

    def insert(self, error, index):
        delta = (error + self.epsilon) ** self.alpha - self.weights.vals[index+self.size-1]
        self.weights.update(delta, index)

    def __len__(self):
        return self.size

    def save(self, folder_name):
        """Save the replay buffer to a folder"""

        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        np.save(folder_name + '/state_old.npy', self.state_old)
        np.save(folder_name + '/actions.npy', self.actions)
        np.save(folder_name + '/rewards.npy', self.rewards)
        np.save(folder_name + '/state_new.npy', self.state_new)
        np.save(folder_name + '/rewards_maintenance.npy', self.rewards_maintenance)

    def load(self, folder_name):
        """Loads the replay buffer from a folder"""
        self.state_old = np.load(folder_name + '/state_old.npy')
        self.actions = np.load(folder_name + '/actions.npy')
        self.rewards = np.load(folder_name + '/rewards.npy')
        self.state_new = np.load(folder_name + '/state_new.npy')
        self.rewards_maintenance = np.load(folder_name + '/rewards_maintenance.npy')

class BDQN_Agent():
    def __init__(self,
                 state_size,
                 action_size,
                 Main_BDQN,
                 Target_BDQN,
                 Replay_buffer,
                 replay_buffer_start_size,
                 gamma=0.99,
                 batch_size=32):
        # state vector and action space
        self.state_size = state_size
        self.action_size = action_size
        # Define DDPG network
        self.main_BDQN = Main_BDQN
        self.target_BDQN = Target_BDQN

        self.eps_evaluation = 0.0
        self.Replay_buffer = Replay_buffer
        self.replay_buffer_start_size = replay_buffer_start_size
        self.gamma = gamma
        self.batch_size = batch_size

    def calc_epsilon(self, count, evaluation=False):

        if evaluation:
            return self.eps_evaluation
        elif count <= 300000:
            amplitude = 1 - 0.9 / 300000 * count
        elif count > 300000 and count <= 2900000:
            amplitude = 1 - 0.9 - 0.09 / 2600000 * (count - 300000)
        else:
            amplitude = 0.01

        return amplitude

    def get_action(self, state, number_iteration, num_action_branch=14, eps=0):

        # calculate the epsilon to choose random action
        eps = self.calc_epsilon(number_iteration)

        Q_value = self.main_BDQN(state).numpy()
        argmax = tf.argmax(Q_value, axis=2).numpy()
        argmax = argmax.reshape(-1)

        output_actions = np.zeros((num_action_branch,), dtype=int)

        for dim in range(num_action_branch):
            deterministic_action = argmax[dim]
            chose_random = np.random.rand(1)
            if chose_random < eps:
                random_action = tf.random.uniform([], minval=0, maxval=3, dtype=tf.int64).numpy()
                output_actions[dim] = random_action
            else:
                output_actions[dim] = deterministic_action

        return output_actions

    def update_target_network(self, Main_network, Target_network):
        """Update the target Q network"""
        Target_network.set_weights(Main_network.get_weights())

    def add_experience(self, action, state_old, reward, state_new, normalized_Reward_maintenance):
        """Wrapper function for adding an experience to the Agent's replay buffer"""
        self.Replay_buffer.add_experience(self, action, state_old, reward, state_new, normalized_Reward_maintenance)

    def learn(self, beta):
        """Sample a batch and use it to improve the BDQN"""
        states, actions, rewards, new_states, reward_maintenance, indices, probs = self.Replay_buffer.get_minibatch(
            batch_size=self.batch_size)
        states, actions, rewards, new_states, reward_maintenance, indices, probs = np.stack(states), np.stack(
            actions), np.stack(
            rewards).reshape(-1, 1), np.stack(new_states), np.stack(reward_maintenance), np.stack(indices), np.stack(
            probs)

        w = 1 / len(self.Replay_buffer) / probs  # weight
        w = w / np.max(w)  # normalized

        # calculate the beta to adjust the learning rate between entire network and branch

        optimizer1 = tf.optimizers.Adam(1e-5)

        # Main DQN estimates best action in new states
        Main_Q_value_new = self.main_BDQN(new_states)
        actions_argmax = tf.argmax(Main_Q_value_new, axis=2)

        # Target DQN estimates q-vals for new states
        future_Q_value = self.target_BDQN(new_states)
        double_q = tf.gather_nd(future_Q_value,
                                [[[index_1, index_2, j] for index_2, j in enumerate(i)] for index_1, i in
                                 enumerate(actions_argmax)])

        # Calculate targets (bellman equation) r + gamma * \sum max_Q(s',a'|theta') /N
        expected_rewards = rewards + (self.gamma * double_q)

        with tf.GradientTape() as tape:
            q_values = self.main_BDQN(states)
            estimate_rewards = tf.gather_nd(q_values,
                                            [[[index_1, index_2, j] for index_2, j in enumerate(i)] for index_1, i in
                                             enumerate(actions)])
            dim_TD_error = estimate_rewards - expected_rewards
            dim_loss = tf.square(dim_TD_error)
            dim_loss = tf.reduce_mean(dim_loss, axis=1)
            mean_loss = tf.keras.losses.Huber()(estimate_rewards, expected_rewards, np.array(w).reshape(1,-1,1))
        model_gradients = tape.gradient(mean_loss, self.main_BDQN.trainable_variables)
        optimizer1.apply_gradients(zip(model_gradients, self.main_BDQN.trainable_variables))

        # branch loss
        expected_branch_rewards = reward_maintenance + (self.gamma * double_q)
        for i in range(self.action_size):
            branch_layer1 = 'branch_layer1_'+str(i)
            branch_layer2 = 'branch_layer2_'+str(i)
            branch_layer3 = 'branch_layer3_' + str(i)
            train_var_list = self.train_variables(self.main_BDQN, branch_layer1, branch_layer2, branch_layer3)
            optimizer2 = tf.optimizers.Adam(1e-5 * beta[i])
            with tf.GradientTape() as tapes:
                q_values = self.main_BDQN(states)
                estimate_rewards = tf.gather_nd(q_values,
                                                [[[index_1, index_2, j] for index_2, j in enumerate(i)] for index_1, i
                                                 in
                                                 enumerate(actions)])
                branch_dim_loss = tf.keras.losses.Huber()(expected_branch_rewards[:, i], estimate_rewards[:, i])
                # branch_dim_loss = tf.reduce_mean(branch_dim_loss, axis=1)
                branch_mean_loss = tf.reduce_mean(branch_dim_loss)
            model_gradients = tapes.gradient(branch_mean_loss, train_var_list)
            optimizer2.apply_gradients(zip(model_gradients, train_var_list))

        dim_TD = dim_loss
        dim_TD = tf.abs(tf.reshape(dim_TD, [-1]))
        for i in range(self.batch_size):
            self.Replay_buffer.insert(dim_TD[i].numpy(), indices[i])  # 更新td error
        return float(mean_loss.numpy())
    def save(self,folder_name):
        """
            Args:
            folder_name: Folder in which to save the Agent
        """
        # Create the folder for saving the agent
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)

        # Save Main_DQN and Target_DQN
        self.main_BDQN.save_weights(folder_name + '/main_BDQN.ckpt')
        self.target_BDQN.save_weights(folder_name + '/target_BDQN.ckpt')

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
        self.main_BDQN = tf.keras.models.load_model(folder_name + '/main_BDQN.ckpt')
        self.target_BDQN = tf.keras.models.load_model(folder_name + '/target_BDQN.ckpt')

        # Load replay buffer
        if load_replay_buffer:
            self.Replay_buffer.load(folder_name + '/Replay-buffer')

        if load_replay_buffer:
            with open('data.txt', 'r+') as f:
                lines = f.readlines()[0].split(" ")

            self.Replay_buffer.count = int(lines[0])
            self.Replay_buffer.current = int(lines[1])

    def train_variables(self,trainModel, branch_layer1, branch_layer2, branch_layer3):
        train_variable_list = []
        for var in trainModel.trainable_variables:
            layers_name = var.name.split("/")
            if (branch_layer1 in layers_name) or (branch_layer2 in layers_name) or (branch_layer3 in layers_name):
                train_variable_list.append(var)
        return train_variable_list
def main():
    Environment = environment()
    initial_states = Environment.reset()
    Main_BDQN = BDQN(70, 14)
    Target_BDQN = BDQN(70, 14)

    Replay_buffer = ReplayBuffer()
    num_episode = 30000
    max_over_step = 100
    batch_size = 32
    replay_buffer_start_size = 1000
    gamma = 0.99
    state_number = 5
    initial_hidden_state = np.array([[0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0]])
    count = 1
    component_number = 14
    agent = BDQN_Agent(70, 14, Main_BDQN, Target_BDQN, Replay_buffer,
                       replay_buffer_start_size, gamma=gamma, batch_size=batch_size)

    beta = np.ones((component_number,))/0.9
    reward_sum = 1
    Branch_reward = np.ones((component_number,))
    reward_plot = []

    for i in range(0, num_episode):
        t = 0
        states = initial_states
        hidden_state = initial_hidden_state

        beta = 0.99 * beta + 0.01 * (Branch_reward / reward_sum)
        beta_decay = beta * (0.99997 ** i)
        reward_sum = 0
        Branch_reward = np.zeros((component_number,))

        # action_episode_1 = np.zeros((101,))
        # action_episode_2 = np.zeros((101,))
        # action_episode_3 = np.zeros((101,))
        # action_episode_4 = np.zeros((101,))
        # action_episode_5 = np.zeros((101,))
        # action_episode_6 = np.zeros((101,))
        # action_episode_7 = np.zeros((101,))
        # hidden_state_episode_1 = np.zeros((101,))
        # hidden_state_episode_2 = np.zeros((101,))
        # hidden_state_episode_3 = np.zeros((101,))
        # hidden_state_episode_4 = np.zeros((101,))
        # hidden_state_episode_5 = np.zeros((101,))
        # hidden_state_episode_6 = np.zeros((101,))
        # hidden_state_episode_7 = np.zeros((101,))
        while t <= max_over_step:

            action = agent.get_action(states, count, component_number)

            count += 1
            #show the last step states
            Hidden_state = hidden_state
            # hidden_state_episode_1[t] = hidden_state[0, 0]
            # hidden_state_episode_2[t] = hidden_state[0, 2]
            # hidden_state_episode_3[t] = hidden_state[0, 3]
            # hidden_state_episode_4[t] = hidden_state[0, 4]
            # hidden_state_episode_5[t] = hidden_state[0, 7]
            # hidden_state_episode_6[t] = hidden_state[0, 10]
            # hidden_state_episode_7[t] = hidden_state[0, 12]
            #
            # action_episode_1[t] = action[0]
            # action_episode_2[t] = action[2]
            # action_episode_3[t] = action[3]
            # action_episode_4[t] = action[4]
            # action_episode_5[t] = action[7]
            # action_episode_6[t] = action[10]
            # action_episode_7[t] = action[12]

            t += 1

            New_belief_state, rewards, hidden_state, normalized_Reward_maintenance = Environment.step(states, action, hidden_state)
            Replay_buffer.add_experience(action, states, rewards, New_belief_state, normalized_Reward_maintenance)

            states = New_belief_state
            reward_sum = reward_sum + rewards
            Branch_reward =  Branch_reward + normalized_Reward_maintenance

            if t % 4 == 0 and Replay_buffer.count >= agent.replay_buffer_start_size - 1:
                loss = agent.learn(beta_decay)
                agent.update_target_network(Main_BDQN, Target_BDQN)

                if t == max_over_step:
                    print("epoch num:", i, " time step: ", t, "   loss: ", loss, "   Reward: ", reward_sum)
                    print("action:       ", action)
                    print("hidden_state:", Hidden_state)
                    print("Beta", np.round(beta_decay, 2))
                    print("------------------------------------------------------------------------------------------")
                    reward_plot.append(reward_sum)
                    # plt.ion()
                    # plt.figure(1, figsize=(4, 2))
                    # plt.clf()
                    # plt.plot(np.linspace(0, 101, 101), action_episode_7)
                    # plt.plot(np.linspace(0, 101, 101), hidden_state_episode_7)
                    # plt.draw()
                    # plt.pause(0.1)
                    # plt.figure(2, figsize=(4, 2))
                    # plt.clf()
                    # plt.plot(np.linspace(0, 101, 101), action_episode_2)
                    # plt.plot(np.linspace(0, 101, 101), hidden_state_episode_2)
                    # plt.draw()
                    # plt.pause(0.1)
                    # plt.figure(3, figsize=(4, 2))
                    # plt.clf()
                    # plt.plot(np.linspace(0, 101, 101), action_episode_3)
                    # plt.plot(np.linspace(0, 101, 101), hidden_state_episode_3)
                    # plt.draw()
                    # plt.pause(0.1)
                    # plt.figure(4, figsize=(4, 2))
                    # plt.clf()
                    # plt.plot(np.linspace(0, 101, 101), action_episode_4)
                    # plt.plot(np.linspace(0, 101, 101), hidden_state_episode_4)
                    # plt.draw()
                    # plt.pause(0.1)
                    # plt.figure(5, figsize=(4, 2))
                    # plt.clf()
                    # plt.plot(np.linspace(0, 101, 101), action_episode_5)
                    # plt.plot(np.linspace(0, 101, 101), hidden_state_episode_5)
                    # plt.draw()
                    # plt.pause(0.1)
                    # plt.figure(6, figsize=(4, 2))
                    # plt.clf()
                    # plt.plot(np.linspace(0, 101, 101), action_episode_6)
                    # plt.plot(np.linspace(0, 101, 101), hidden_state_episode_6)
                    # plt.draw()
                    # plt.pause(0.1)
                    # plt.figure(7, figsize=(4, 2))
                    # plt.clf()
                    # plt.plot(np.linspace(0, 101, 101), action_episode_7)
                    # plt.plot(np.linspace(0, 101, 101), hidden_state_episode_7)
                    # plt.draw()
                    # plt.pause(0.1)
    np.savetxt('test.out', reward_plot)
    agent.save('dqnmodel')

if "__main__" == __name__:
    main()

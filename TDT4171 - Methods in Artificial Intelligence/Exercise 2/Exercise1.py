"""
This file is included simply because I used it for some calculations in WRONGexercise2, before I realized it would
be easier to simply create a HMM-class and call functions from there.

It has not been changed since Exercise 1, and is barely used in Exercise 2.

Seriously, there's nothing to see here!
"""

from collections import defaultdict

import numpy as np


class Variable:
    def __init__(self, name : str, no_states : int, table : list, parents=[], no_parent_states=[]):
        """
        name (string): Name of the variable
        no_states (int): Number of states this variable can take
        table (list or Array of reals): Conditional probability table (see below)
        parents (list of strings): Name for each parent variable.
        no_parent_states (list of ints): Number of states that each parent variable can take.

        The table is a 2d array of size #events * #number_of_conditions.
        #number_of_conditions is the number of possible conditions (prod(no_parent_states))
        If the distribution is unconditional #number_of_conditions is 1.
        Each column represents a conditional distribution and sum to 1.

        Here is an example of a variable with 3 states and two parents cond0 and cond1,
        with 3 and 2 possible states respectively.
        +----------+----------+----------+----------+----------+----------+----------+
        |  cond0   | cond0(0) | cond0(1) | cond0(2) | cond0(0) | cond0(1) | cond0(2) |
        +----------+----------+----------+----------+----------+----------+----------+
        |  cond1   | cond1(0) | cond1(0) | cond1(0) | cond1(1) | cond1(1) | cond1(1) |
        +----------+----------+----------+----------+----------+----------+----------+
        | event(0) |  0.2000  |  0.2000  |  0.7000  |  0.0000  |  0.2000  |  0.4000  |
        +----------+----------+----------+----------+----------+----------+----------+
        | event(1) |  0.3000  |  0.8000  |  0.2000  |  0.0000  |  0.2000  |  0.4000  |
        +----------+----------+----------+----------+----------+----------+----------+
        | event(2) |  0.5000  |  0.0000  |  0.1000  |  1.0000  |  0.6000  |  0.2000  |
        +----------+----------+----------+----------+----------+----------+----------+

        To create this table you would use the following parameters:

        Variable('event', 3, [[0.2, 0.2, 0.7, 0.0, 0.2, 0.4],
                              [0.3, 0.8, 0.2, 0.0, 0.2, 0.4],
                              [0.5, 0.0, 0.1, 1.0, 0.6, 0.2]],
                 parents=['cond0', 'cond1'],
                 no_parent_states=[3, 2])
        """
        self.name = name
        self.no_states = no_states
        self.table = np.array(table)
        self.parents = parents
        self.no_parent_states = no_parent_states

        if self.table.shape[0] != self.no_states:
            raise ValueError(f"Number of states and number of rows in table must be equal. "
                             f"Recieved {self.no_states} number of states, but table has "
                             f"{self.table.shape[0]} number of rows.")

        if self.table.shape[1] != np.prod(no_parent_states):
            raise ValueError("Number of table columns does not match number of parent states combinations.")

        if not np.allclose(self.table.sum(axis=0), 1):
            raise ValueError("All columns in table must sum to 1.")

        if len(parents) != len(no_parent_states):
            raise ValueError("Number of parents must match number of length of list no_parent_states.")

    def __str__(self) -> str:
        """
        Pretty string for the table distribution
        For printing to display properly, don't use variable names with more than 7 characters
        """
        width = int(np.prod(self.no_parent_states))
        grid = np.meshgrid(*[range(i) for i in self.no_parent_states])
        s = ""
        for (i, e) in enumerate(self.parents):
            s += '+----------+' + '----------+' * width + '\n'
            gi = grid[i].reshape(-1)
            s += f'|{e:^10}|' + '|'.join([f'{e + "("+str(j)+")":^10}' for j in gi])
            s += '|\n'

        for i in range(self.no_states):
            s += '+----------+' + '----------+' * width + '\n'
            state_name = self.name + f'({i})'
            s += f'|{state_name:^10}|' + '|'.join([f'{p:^10.4f}' for p in self.table[i]])
            s += '|\n'

        s += '+----------+' + '----------+' * width + '\n'

        return s

    def probability(self, state : int, parentstates : dict) -> float:
        """
        Returns probability of variable taking on a "state" given "parentstates"
        This method is a simple lookup in the conditional probability table, it does not calculate anything.

        Input:
            state: integer between 0 and no_states
            parentstates: dictionary of {'parent': state}
        Output:
            float with value between 0 and 1
        """
        if not isinstance(state, int):
            raise TypeError(f"Expected state to be of type int; got type {type(state)}.")
        if not isinstance(parentstates, dict):
            raise TypeError(f"Expected parentstates to be of type dict; got type {type(parentstates)}.")
        if state >= self.no_states:
            raise ValueError(f"Recieved state={state}; this variable's last state is {self.no_states - 1}.")
        if state < 0:
            raise ValueError(f"Recieved state={state}; state cannot be negative.")

        table_index = 0
        for variable in self.parents:
            if variable not in parentstates:
                raise ValueError(f"Variable {variable.name} does not have a defined value in parentstates.")

            var_index = self.parents.index(variable)
            table_index += parentstates[variable] * np.prod(self.no_parent_states[:var_index])

        return self.table[state, int(table_index)]


class BayesianNetwork:
    """
    Class representing a Bayesian network.
    Nodes can be accessed through self.variables['variable_name'].
    Each node is a Variable.

    Edges are stored in a dictionary. A node's children can be accessed by
    self.edges[variable]. Both the key and value in this dictionary is a Variable.
    """
    def __init__(self):
        self.edges = defaultdict(lambda: [])  # All nodes start out with 0 edges
        self.variables = {}                   # Dictionary of "name":TabularDistribution

    def add_variable(self, variable : 'Variable'):
        """
        Adds a variable to the network.
        """
        if not isinstance(variable, Variable):
            raise TypeError(f"Expected {Variable}; got {type(variable)}.")
        self.variables[variable.name] = variable

    def add_edge(self, from_variable : 'Variable', to_variable : 'Variable'):
        """
        Adds an edge from one variable to another in the network. Both variables must have
        been added to the network before calling this method.
        """
        if from_variable not in self.variables.values():
            raise ValueError("Parent variable is not added to list of variables.")
        if to_variable not in self.variables.values():
            raise ValueError("Child variable is not added to list of variables.")
        self.edges[from_variable].append(to_variable)

    def sorted_nodes(self) -> list:
        """
        An implementation of Kahn's algorithm, as described in Wikipedia
            ( https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm )
        Returns: List of sorted variable names.
        """
        sorted_list = []    # L in the pseudocode
        edges = self.edges.copy()
        unsorted_variables = self.variables.copy()
        nodes_without_incoming = [] # S in the pseudocode

        def has_incoming_edge(node):
            """Iterates through edges.values to see if node has incoming edges"""
            for to_nodes in edges.values():
                if node in to_nodes:
                    # Is a to_variable => Has incoming edge
                    return True
            return False    

        def move_node_without_incoming(node):
            """Moves a node from unsorted_variables to nodes_without_incoming"""
            nodes_without_incoming.append(node)
            unsorted_variables.pop(node.name)

        # Iterate through remaining values, move nodes without incoming edge to nodes_without_incoming
        nodes_to_be_moved = [] # Used to avoid changing size of unsorted_variables while iterating
        for node in unsorted_variables.values():
            if has_incoming_edge(node) == False:
                nodes_to_be_moved.append(node)
        for node in nodes_to_be_moved:
            move_node_without_incoming(node)

        while len(nodes_without_incoming) > 0:
            temp_node = nodes_without_incoming.pop()
            sorted_list.append(temp_node.name)
            if temp_node in edges:
                # There exists edges going out from temp_node
                edge = edges.pop(temp_node)
                for receiving_node in edge:
                    if has_incoming_edge(receiving_node) == False:
                        # receiving_node has no other edges going into it
                        move_node_without_incoming(receiving_node)

        if len(edges) > 0:
            raise RuntimeError(f"After sorting, graph has {len(edges)} edges, and therefore has a loop.")
        return sorted_list



class InferenceByEnumeration:
    def __init__(self, bayesian_network):
        self.bayesian_network = bayesian_network
        self.topo_order = bayesian_network.sorted_nodes()

    def _enumeration_ask(self, X : str, evidence):
        """Implementation of the ENUMERATE-ASK algorithm found in figure 14.9 in Russel & Norvig (p. 525)
        
        Arguments:
            - X        : Name of variable to find distribution over
            - evidence : dictionary of {variable name: variable value}

        Returns:
            - the distribution over X
        """
        number_of_X_states = self.bayesian_network.variables.get(X).no_states
        Q = [None] * number_of_X_states
        for possible_state in range(number_of_X_states):
            temp_evidence = evidence.copy()
            temp_evidence[X] = possible_state
            Q[possible_state] = self._enumerate_all(self.topo_order, temp_evidence)

        return [float(i)/sum(Q) for i in Q] # Normalizes Q


    def _enumerate_all(self, vars : list, evidence : dict) -> float:
        """Recursive implementation of the ENUMERATE-ALL algorithm found in figure 14.9 in Russel & Norvig (p. 525)

        Arguments:
            - vars     : topologically ordered list of variable names
            - evidence : dictionary of {variable name: variable value}
        
        Returns:
            - probability of evidence for the variables in vars
        """
        if len(vars) == 0:
            return 1.0

        Y_name = vars[0]  # Y <-- FIRST(vars)
        Y = self.bayesian_network.variables.get(Y_name)

        # Find parents' values
        parent_values = {}
        for parent in Y.parents:
            if parent not in evidence.keys():
                raise ValueError(f"Variable {parent} (parent of {Y.name}) does not have a defined value in evidence.")
            parent_values[parent] = evidence.get(parent)
    
        if Y_name in evidence.keys():
            probability = Y.probability(evidence[Y.name], parent_values)
            return probability * self._enumerate_all(vars[1:], evidence)
        else:
            sum_to_be_returned = 0.0
            for possible_state in range(Y.no_states):
                temp_evidence = evidence.copy()
                temp_evidence[Y_name] = possible_state
                probability = Y.probability(possible_state, parent_values)
                sum_to_be_returned += probability * self._enumerate_all(vars[1:], temp_evidence)
            return sum_to_be_returned


    def query(self, var, evidence={}):
        """
        Wrapper around "_enumeration_ask" that returns a
        Tabular variable instead of a vector
        """
        q = np.reshape(self._enumeration_ask(var, evidence),(-1, 1))
        return Variable(var, self.bayesian_network.variables[var].no_states, q)


def problem3c():
    d1 = Variable('A', 2, [[0.8], [0.2]])
    d2 = Variable('B', 2, [[0.5, 0.2],
                           [0.5, 0.8]],
                  parents=['A'],
                  no_parent_states=[2])
    d3 = Variable('C', 2, [[0.1, 0.3],
                           [0.9, 0.7]],
                  parents=['B'],
                  no_parent_states=[2])
    d4 = Variable('D', 2, [[0.6, 0.8],
                           [0.4, 0.2]],
                  parents=['B'],
                  no_parent_states=[2])

    print(f"Probability distribution, P({d1.name})")
    print(d1)

    print(f"Probability distribution, P({d2.name} | {d1.name})")
    print(d2)

    print(f"Probability distribution, P({d3.name} | {d2.name})")
    print(d3)

    print(f"Probability distribution, P({d4.name} | {d2.name})")
    print(d4)

    bn = BayesianNetwork()

    bn.add_variable(d1)
    bn.add_variable(d2)
    bn.add_variable(d3)
    bn.add_variable(d4)
    bn.add_edge(d1, d2)
    bn.add_edge(d2, d3)
    bn.add_edge(d2, d4)

    inference = InferenceByEnumeration(bn)
    posterior = inference.query('C', {'D': 1})

    print(f"Probability distribution, P({d3.name} | !{d4.name})")
    print(posterior)


def monty_hall():
    """Implementation of the Monty Hall problem, as described in task 4c
    
    NOTE: Doors are numbered 0-2, instead of 1-3 (as given in the task)"""
    ChosenByGuest = Variable("Guest", 3, [[1/3], [1/3], [1/3]])
    Prize = Variable("Prize", 3, [[1/3], [1/3], [1/3]])
    OpenedByHost = Variable("Host", 3, [[  0,    0,    0,    0,  1/2,    1,    0,    1,  1/2],
                                        [1/2,    0,    1,    0,    0,    0,    1,    0,  1/2],
                                        [1/2,    1,    0,    1,  1/2,    0,    0,    0,    0]],
                            ["Guest", "Prize"], [3, 3])
    
    network = BayesianNetwork()

    network.add_variable(ChosenByGuest)
    network.add_variable(Prize)
    network.add_variable(OpenedByHost)

    network.add_edge(ChosenByGuest, OpenedByHost)
    network.add_edge(Prize, OpenedByHost)

    inference = InferenceByEnumeration(network)
    posterior = inference.query("Prize", {"Guest": 0, "Host": 2})

    print(f"Probability distribution, P({Prize.name} | ChosenByGuest = 0, OpenedByHost = 2)")
    print(posterior)

if __name__ == '__main__':
    problem3c()
    monty_hall()

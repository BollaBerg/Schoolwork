"""
This file contains the actual solutions to Assignment 2!

As you might see, I am somewhat lacking in comments. I have tried making up for it with (hopefully) understandable
variable names, and I have followed the algorithms found in Russell & Norvig as well as I could. I did include comments
pretty much everywhere in WRONGexercise2, but that was also maybe the only good thing that python-file provided.

I simply ran out of time for this one, and for that I am sorry!
"""
from typing import List
import numpy as np
import matplotlib.pyplot as plt

class HMM:
    def __init__(self,
                 initial_probabilities : List[float],
                 transition_table : List[List[float]],
                 observation_table : List[List[float]],
                 evidence : List[int]):
        self.initial = initial_probabilities

        self.transition_table = transition_table
        # transition_table[prev][curr] = P(X{curr} | X{prev})
        self.observation_table = observation_table
        # observation_table[state][obs] = P(e = obs | X = state)
        self.evidence = evidence

        self.number_of_states = len(transition_table)
        self.number_of_observation_states = len(observation_table[0])

    def normalize(self, lst):
        _sum = sum(lst)
        _factor = 1/_sum
        return [_factor * elem for elem in lst]

    def filter(self):
        previous = self.initial
        _output = []

        for t in range(len(self.evidence)):
            _prediction = self.predict_next(previous)

            probability_of_evidence = [self.observation_table[state][self.evidence[t]]\
                                        for state in range(self.number_of_states)]

            probability = [a*b for (a,b) in zip(probability_of_evidence, _prediction)]

            normal_probability = self.normalize(probability)
            _output.append(normal_probability)
            previous = normal_probability
        
        return _output


    def predict_next(self, prev):
        _sum = [0] * self.number_of_states
        for state in range(self.number_of_states):
            prob_next_state = self.transition_table[state]
            _result = [prev[state] * elem for elem in prob_next_state]
            _sum = [a + b for (a,b) in zip(_sum, _result)]
        return _sum

    def predict(self, previous, number):
        output = []
        for _ in range(number):
            _result = self.predict_next(previous)
            output.append(_result)
            previous = _result

        return output

    def backward_next(self, prev, t):
        _sum = [0] * self.number_of_states
        for prev_state in range(self.number_of_states):
            prob_evidence = self.observation_table[prev_state][self.evidence[t]]
            prob_prev = prev[prev_state]
            prob_state_if_prev = [self.transition_table[state][prev_state] for state in range(self.number_of_states)]
            first_two_factors = prob_evidence * prob_prev
            _line = [first_two_factors*elem for elem in prob_state_if_prev]
            _sum = [a+b for (a,b) in zip(_sum, _line)]
        return _sum

    def smooth(self):
        output = [] 
        backwards = []
        backwards_prev = [1]*self.number_of_states

        for t in range(len(self.evidence)-1, -1, -1):
            _result = self.backward_next(backwards_prev, t)
            backwards.insert(0, _result)
            backwards_prev = _result
        
        forwards = self.filter()
        forwards.insert(0, self.initial)

        for t in range(len(self.evidence)):
            line = [a*b for (a,b) in zip(backwards[t], forwards[t])]
            result = self.normalize(line)
            output.append(result)

        return output

    def most_likely_sequence(self, t):
        previous = max(self.initial)
        _probabilities = []

        for t in range(len(self.evidence)):
            outer_probability = [self.observation_table[i][self.evidence[t]] for i in range(len(self.observation_table))]
            max_probability = [0]*self.number_of_states
            for prev_state in range(self.number_of_states):
                _probability = self.transition_table[prev_state]
                probability = [elem*previous for elem in _probability]

def task_1b():
    result_b = hmm.filter()
    for i in range(1, 7):
        print(F"Probability distribution, P(X{i} | e{{1:{i}}}):")
        for index, state in enumerate(["No fish", "Fish"]):
            key_state = F"X_{i}({state})"
            probability = F"{result_b[i-1][index]:.4f}"

            print(header)
            print(F"|{key_state.center(key_length)}|{probability.center(key_length)}|")
        print(header)
        print()

def task_1c():
    result_c = hmm.predict(hmm.filter()[-1], (31-7)) 
    for i in range(7, 31):
        print(F"Probability distribution, P(X{i} | e{{1:6}}):")
        for index, state in enumerate(["No fish", "Fish"]):
            key_state = F"X_{i}({state})"
            probability = F"{result_c[i-7][index]:.4f}"

            print(header)
            print(F"|{key_state.center(key_length)}|{probability.center(key_length)}|")
        print(header)
        print()

    # Plot everything
    x_axis = [i for i in range(7, 31)]
    y_axis = [i[1] for i in result_c]

    figure, axes = plt.subplots()
    axes.plot(x_axis, y_axis)
    axes.set(xlabel = 't', ylabel = 'P(X_t = Fish)',
            title = 'Plot of task 1c')
    
    figure.savefig('Task 1c.png')

def task_1d():
    result_d = hmm.smooth()
    for i in range(6):
        print(F"Probability distribution, P(X{i} | e{{1:6}}):")
        for index, state in enumerate(["No fish", "Fish"]):
            key_state = F"X_{i}({state})"
            probability = F"{result_d[i][index]:.4f}"

            print(header)
            print(F"|{key_state.center(key_length)}|{probability.center(key_length)}|")
        print(header)
        print()

def task_1e():
    from WRONGexercise2 import task1e
    task1e()

if __name__ == '__main__':
    initial_probabilities = [0.5, 0.5]
    transition_table = [[0.7, 0.3],
                        [0.2, 0.8]]
    observation_table = [[0.8, 0.2],
                         [0.25, 0.75]]
    evidence = [1, 1, 0, 1, 0, 1]

    hmm = HMM(initial_probabilities, transition_table, observation_table, evidence)

    key_length = 15
    header = F"+{'-'*key_length}+{'-'*key_length}+"

    task_1b()
    task_1c()
    task_1d()
    task_1e()
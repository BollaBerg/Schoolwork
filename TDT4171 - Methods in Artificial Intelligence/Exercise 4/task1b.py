import math

import pandas as pd
import numpy as np

def plurality_value(examples):
    """Return most common output value among examples"""
    most_likely_result = examples.Survived.mode()[0]
    return most_likely_result


def argmax_dict(dictionary):
    """Return index of highest value in dictionary
    
    Assumes dictionary has max-able values (such as numbers)
    """
    f = lambda i: dictionary[i]
    return max(dictionary.keys(), key=f)


def boolean_entropy(val):
    if val in [0, 1]:
        return 0
    return -(val * math.log(val, 2) + (1-val) * math.log((1-val), 2))


class Node:
    def __init__(self, label):
        self.label = label
        self.children = dict()

    def add_parent(self, parent, parent_value):
        self.parent = parent
        parent.children[parent_value] = self

    def __repr__(self): return F"Node({self.label})"


class Tree:
    def __init__(self, root_label):
        self.root = Node(root_label)
    
    def add_branch(self, subtree, parent_value):
        subtree.root.add_parent(self.root, parent_value)

    def print_nicely(self):
        def dfs_print(node, level, parent_value):
            print(level*"----" + str(parent_value) + "---" + str(node.label))
            for value, child in node.children.items():
                dfs_print(child, level+1, value)
        
        print(self.root)
        for value, child in self.root.children.items():
            dfs_print(child, 1, value)



class DecisionTree:
    UNUSED_ATTRIBUTES = [
        "Survived", "Name", "Ticket", "Cabin"
    ]
    CONTINUOUS_ATTRIBUTES = [
        "Age", "Fare"
    ]

    def __init__(self, filename):
        self.dataframe = pd.read_csv(filename)
        self.values_of_attribute = {}
        for column in self.dataframe:
            self.values_of_attribute[column] = self.dataframe[column].unique()
        attributes = self.values_of_attribute.keys()

        self.attributes = [a for a in attributes if
                            a not in self.UNUSED_ATTRIBUTES]

        self.curr_max_split = {}

    def learn(self):
        self.tree = self.decision_tree_learning(
                                self.dataframe,
                                self.attributes,
                                [])
        return self.tree

    def predict(self, row):
        node = self.tree.root
        while len(node.children) > 0:
            row_value = row[node.label]
            if node.label in self.CONTINUOUS_ATTRIBUTES:
                string = str([k for k in node.children.keys()][0])
                compare_to = float(string[2:])
                if row_value <= compare_to:
                    node = [n for n in node.children.values()][0]
                else:
                    node = [n for n in node.children.values()][1]
            else:
                node = node.children[row_value]
        return node.label

    def run(self, filename):
        testframe = pd.read_csv(filename)
        fails = runs = 0
        for _, row in testframe.iterrows():
            runs += 1
            predicted_result = self.predict(row)
            if predicted_result != row["Survived"]:
                fails += 1
        
        return fails, runs


    def decision_tree_learning(self, examples, attributes, parent_examples):
        if len(examples) == 0:
            return Tree(plurality_value(parent_examples))
        
        if len(np.unique(examples["Survived"])) == 1:
            return Tree(examples["Survived"].iloc[0])
        
        if attributes == None or len(attributes) == 0:
            return Tree(plurality_value(examples))

        A = argmax_dict({a : self.importance(a, examples) for a in attributes})
        tree = Tree(A)
        attributes_without_A = [a for a in attributes if a != A]
        if A in self.CONTINUOUS_ATTRIBUTES:
            max_split = self.curr_max_split[A]
            exs_low = examples[examples[A] <= max_split]
            subtree = self.decision_tree_learning(exs_low,
                                                  attributes_without_A,
                                                  examples)
            tree.add_branch(subtree, F"<={max_split}")
            exs_high = examples[examples[A] > max_split]
            subtree = self.decision_tree_learning(exs_high,
                                                  attributes_without_A,
                                                  examples)
            tree.add_branch(subtree, F"> {max_split}")
            return tree
        
        for value in self.values_of_attribute[A]:
            if value == 'nan':
                exs = examples[examples[A].isna()]
            exs = examples[examples[A] == value]
            subtree = self.decision_tree_learning(exs,
                                                  attributes_without_A,
                                                  examples)
            tree.add_branch(subtree, value)
        return tree
        

    def importance(self, attribute, examples):
        if attribute in self.CONTINUOUS_ATTRIBUTES:
            return self.importance_cont(attribute, examples)
        total_count = examples.Survived.value_counts()
        negatives_all = total_count.get(0, default=0)
        positives_all = total_count.get(1, default=0)
        inside_B = positives_all / (positives_all + negatives_all)
        base_entropy = boolean_entropy(inside_B)

        remainder = 0
        for value in self.values_of_attribute[attribute]:
            entries = (examples[examples[attribute] == value]
                            ["Survived"].value_counts())
            negatives = entries.get(0, default=0)
            positives = entries.get(1, default=0)

            if positives == 0:
                continue

            value_inside_B = positives / (positives + negatives)
            p = (positives + negatives) / (positives_all + negatives_all)
            remainder += p * boolean_entropy(value_inside_B)
        
        gain = base_entropy - remainder
        return gain

    def importance_cont(self, attribute, examples):
        values = self.values_of_attribute[attribute]
        total_count = examples.Survived.value_counts()
        negatives_all = total_count.get(0, default=0)
        positives_all = total_count.get(1, default=0)
        inside_B = positives_all / (positives_all + negatives_all)
        base_entropy = boolean_entropy(inside_B)

        gains = {}
        for i in range(1, len(values)):
            split = values[i]
            remainder = 0
            entries_below = (examples[examples[attribute] <= split]
                            ["Survived"].value_counts())
            negatives = entries_below.get(0, default=0)
            positives = entries_below.get(1, default=0)

            if positives == 0:
                continue

            value_inside_B = positives / (positives + negatives)
            p = (positives + negatives) / (positives_all + negatives_all)
            remainder += p * boolean_entropy(value_inside_B)

            entries_above = (examples[examples[attribute] > split]
                            ["Survived"].value_counts())
            negatives = entries_above.get(0, default=0)
            positives = entries_above.get(1, default=0)

            if positives == 0:
                continue

            value_inside_B = positives / (positives + negatives)
            p = (positives + negatives) / (positives_all + negatives_all)
            remainder += p * boolean_entropy(value_inside_B)
            
            gain = base_entropy - remainder
            gains[split] = gain
        
        if len(gains) == 0:
            if not self.curr_max_split[attribute]:
                self.curr_max_split[attribute] = 0
            return 0
        max_split = max(gains.keys(), key=lambda k: gains[k])
        self.curr_max_split[attribute] = max_split # BAD way of doing it, but YOLO
        return gains[max_split]


if __name__ == "__main__":
    tree = DecisionTree("data/train.csv")
    tree.learn()
    tree.tree.print_nicely()
    fails, runs = tree.run("data/test.csv")
    accuracy = 1 - (fails / runs)
    print(F"Failed {fails} times on {runs} runs")
    print(F"Accuracy: {accuracy}")
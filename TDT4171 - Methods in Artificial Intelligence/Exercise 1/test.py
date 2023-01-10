from Exercise1 import *

def test_4a():
    network = BayesianNetwork()
    var1 = Variable("var1", 2, [[0.3],
                                [0.7]])
    var2 = Variable("var2", 2, [[0.1, 0.6],
                                [0.9, 0.4]], ["var1"], [2])
    var3 = Variable("var3", 3, [[0.1, 0.4, 0.3, 0.8],
                                [0.3, 0.5, 0.1, 0.0],
                                [0.6, 0.1, 0.6, 0.2]], ["var1", "var2"], [2, 2])

    network.add_variable(var1)
    network.add_variable(var2)
    network.add_variable(var3)
    network.add_edge(var1, var2)
    network.add_edge(var1, var3)
    network.add_edge(var2, var3)

    print(network.sorted_nodes())


def test_using_problem_2():
    network = BayesianNetwork()
    A = Variable("A", 2, [[0.5], [0.5]])
    B = Variable("B", 2, [[0.5, 0.5], [0.5, 0.5]], ["C"], [2])
    C = Variable("C", 2, [[0.5, 0.5], [0.5, 0.5]], ["A"], [2])
    D = Variable("D", 2, [[0.5, 0.5], [0.5, 0.5]], ["H"], [2])
    E = Variable("E", 2, [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]], ["C", "D"], [2, 2])
    F = Variable("F", 2, [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]], ["E", "G"], [2, 2])
    G = Variable("G", 2, [[0.5, 0.5], [0.5, 0.5]], ["H"], [2])
    H = Variable("H", 2, [[0.5], [0.5]])

    network.add_variable(A)
    network.add_variable(B)
    network.add_variable(C)
    network.add_variable(D)
    network.add_variable(E)
    network.add_variable(F)
    network.add_variable(G)
    network.add_variable(H)

    network.add_edge(A, C)
    network.add_edge(C, B)
    network.add_edge(C, E)
    network.add_edge(D, E)
    network.add_edge(E, F)
    network.add_edge(G, F)
    network.add_edge(H, D)
    network.add_edge(H, G)

    print(network.sorted_nodes())

def test_using_problem_3c():
    problem3c()
    print("SHOULD BE: ")
    print("C(0): 0.1778")
    print("C(1): 0.8222")

def test_using_monty_hall():
    monty_hall()
    print("SHOULD BE: ")
    print("Prize(0): 0.3333")
    print("Prize(1): 0.6667")
    print("Prize(2): 0.0000")

if __name__ == '__main__':
    test_4a()
    test_using_problem_2()
    test_using_problem_3c()
    test_using_monty_hall()
    
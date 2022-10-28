import numpy as np
"""
This program is used to illustrate the three basic structures, the chain, fork,
and collider, in a graph model. The two variables X and Y are independent means
that P(X=x, Y=y) = P(X=x) * P(Y=y).
"""


def root_node(t: float, size: int = 1) -> np.ndarray:
    """
    The function returns a binary array of a given size.
    @param: t = P(X = 1)
    """
    x = np.random.random(size)
    x = np.where(x <= t, 1, 0)
    return x


def child_node(x: np.ndarray, t2t: float, f2t: float) -> np.ndarray:
    """
    The function returns a binary array based on the parent node "x".
    @param: t2t = P(Y = 1 | X = 1)
    @param: f2t = P(Y = 1 | X = 0)
    """
    y = np.random.random(x.size)
    y = np.where(np.logical_or(np.logical_and(x == 1, y <= t2t), np.logical_and(x == 0, y <= f2t)), 1, 0)
    return y


def form_table(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    The function returns a three-dimensional array as the statistical result.
    """
    s = np.zeros((2, 2, 2), dtype=np.int64)
    for i, j, k in zip(x, y, z):
        s[i, j, k] += 1
    return s


def print_probability(x: np.ndarray, x_name: str = "X", y_name: str = "Y") -> None:
    """
    The function prints P(x, y) and P(x) * P(y) to show the independence of two variables.
    """
    s = np.sum(x)
    for i in [0, 1]:
        for j in [0, 1]:
            p_x = np.sum(x[i, :]) / s
            p_y = np.sum(x[:, j]) / s
            p_xy = x[i, j] / s
            print("P({0}={1}, {2}={3}) = {4:.4f}, P({0}={1}) * P({2}={3}) = {5:.4f}".format(
                x_name, i, y_name, j, p_xy, p_x * p_y))


if __name__ == '__main__':
    # Chain: X -> Y -> Z
    # Suppose that X means fire emergency, Y means smoke and Z means fire alarm.
    X = root_node(0.02, 10000)
    Y = child_node(X, 0.85, 0.2)
    Z = child_node(Y, 0.98, 0.03)
    S = form_table(X, Y, Z)
    # output results
    print("Chain: X -> Y -> Z")
    print("Raw Data:")
    print(S)
    print("No Condition:")
    print_probability(np.sum(S, axis=1), "X", "Z")
    print("Condition: Y = 0")
    print_probability(S[:, 0, :], "X", "Z")
    print("Condition: Y = 1")
    print_probability(S[:, 1, :], "X", "Z")

    # Fork: X <- Y -> Z
    # Suppose that X means fire emergency, Y means smoke and Z means fire alarm.
    Y = root_node(0.2, 10000)
    X = child_node(Y, 0.9, 0.1)
    Z = child_node(Y, 0.7, 0.2)
    S = form_table(X, Y, Z)
    # output results
    print("Fork: X <- Y -> Z")
    print("Raw Data:")
    print(S)
    print("No Condition:")
    print_probability(np.sum(S, axis=1), "X", "Z")
    print("Condition: Y = 0")
    print_probability(S[:, 0, :], "X", "Z")
    print("Condition: Y = 1")
    print_probability(S[:, 1, :], "X", "Z")

    # Collider: X -> Y <- Z
    # Suppose that X means fire emergency, Y means smoke and Z means fire alarm.
    X = root_node(0.02, 10000)
    Y = root_node(0.85, 10000)
    Z = child_node(Y, 0.98, 0.06)
    S = form_table(X, Y, Z)
    # output results
    print("Collider: X -> Y <- Z")
    print("Raw Data:")
    print(S)
    print("No Condition:")
    print_probability(np.sum(S, axis=1), "X", "Z")
    print("Condition: Y = 0")
    print_probability(S[:, 0, :], "X", "Z")
    print("Condition: Y = 1")
    print_probability(S[:, 1, :], "X", "Z")

import numpy as np


def print_j_p(j_p):
    str_value = ['a', 'b', 'c']
    for i in range(len(j_p)):
        print('\n')
        for j in range(len(j_p[0])):
            print(f"p({str_value[i]}|{str_value[j]}) = {round(j_p[i][j], 3)}")


def is_valid_transition_matrix(p, tol=1e-8):
    if not all(abs(sum(p[:, i]) - 1) < tol for i in range(p.shape[1])):
        print("Each column should sum up to approximately 1.")
        return False
    if not all(0 <= elem <= 1 for elem in p.flatten()):
        print("Each element should be in the range [0, 1].")
        return False
    return True


def get_alt_stat(p):
    n = len(p[0])

    b = np.zeros(n + 1)

    solution = np.linalg.solve(p[:-1], b[:-1])
    if np.isclose(np.sum(solution), b[-1]):
        return solution
    else:
        return None


def get_stat(p):
    if not is_valid_transition_matrix(p):
        print("Invalid transition matrix.")
        return None

    n = len(p[0])

    eq = np.eye(n)
    eq = eq - p
    eq = np.vstack((eq, np.ones_like(p[0])))

    b = np.zeros(n + 1)
    b[n] = 1

    x, residuals, rank, s = np.linalg.lstsq(eq, b, rcond=None)
    return x


def get_h(z):
    non_zero_probs = z[z > 0]
    return -np.sum(non_zero_probs * np.log2(non_zero_probs))


def get_joint_probabilities(p, x):
    j_p = np.zeros_like(p)
    for i in range(len(j_p)):
        for j in range(len(j_p[0])):
            j_p[i][j] = x[j] * p[i][j]
    return j_p


def get_h_conditional(x, j_p):
    joint_probabilities = j_p[:, np.newaxis] * p

    # Calculate the conditional entropy
    entropy_values = joint_probabilities * np.log2(p + np.finfo(float).eps)
    conditional_entropy_value = -np.sum(entropy_values)

    return conditional_entropy_value

if __name__ == '__main__':
    p = np.array([[0.32, 0.06, 0.28],
                  [0.56, 0.73, 0.37],
                  [0.12, 0.21, 0.35]])
    x = get_stat(p)

    h_x = get_h(x)

    j_p = get_joint_probabilities(p, x)

    h_j_p = get_h(j_p)

    h_conditional = h_j_p - h_x

    print('-----------------------')
    print(x)
    print('-----------------------')
    print("\n")

    print('-----------------------')
    print(round(h_x,3))
    print('-----------------------')
    print("\n")

    print('-----------------------')
    print(j_p)
    print_j_p(j_p)
    print('-----------------------')
    print("\n")

    print('-----------------------')
    print(round(h_j_p, 3))
    print('-----------------------')
    print("\n")

    print('-----------------------')
    print(round(h_conditional,3))
    print(get_h_conditional(x, j_p))
    print('-----------------------')
    print("\n")

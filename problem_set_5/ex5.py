import random

class AMSVariable:
    def __init__(self, position, value):
        self.el = position
        self.val = value

def ams_moment(stream, n, k, v):

    variables = []
    for i in range(v):
        pos = random.randint(0, n - 1)
        variables.append(AMSVariable(i, stream[pos]))

    for i, x in enumerate(stream):
        for var in variables:
            if var.el == i:
                var.val = x ** k

    # Compute the k-th moment estimate
    moment_sum = 0
    for var in variables:
        contribution = n * var.val - 1
        moment_sum += contribution

    moment_estimate = moment_sum / v

    return moment_estimate


if __name__ == "__main__":
    stream = [1, 2, 3, 2, 4, 1, 3, 4, 1, 2, 4, 3, 1, 1, 2]
    n = len(stream)
    k = [1, 2, 3] # Degrees of moment
    v = [1, 3, 5, 7, 9]  # Number of auxiliary variables

    for degree in k:
        print(f"\nEstimated {degree}-th moment for:")
        for var in v:
            moment_estimate = ams_moment(stream, n, degree, var)
            print(f"* {var} variable: {moment_estimate}")
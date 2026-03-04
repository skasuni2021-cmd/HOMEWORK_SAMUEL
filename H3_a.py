"""
hw3a.py
MAE 3403 – HW 3
Computes probabilities of a normal distribution using Simpson’s 1/3 rule.
If probability is given, uses the Secant Method to solve for c.
Only uses: math
"""

import math


# ---------------------------------------------------------
# Normal PDF
# ---------------------------------------------------------
def normal_pdf(x, mu, sigma):
    return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)


# ---------------------------------------------------------
# Simpson’s 1/3 Rule
# ---------------------------------------------------------
def simpson(f, a, b, n, mu, sigma):

    if n % 2 == 1:
        n += 1

    h = (b - a) / n
    s = f(a, mu, sigma) + f(b, mu, sigma)

    for i in range(1, n):
        x = a + i * h

        if i % 2 == 0:
            s += 2 * f(x, mu, sigma)
        else:
            s += 4 * f(x, mu, sigma)

    return (h / 3) * s


# ---------------------------------------------------------
# Secant Method
# ---------------------------------------------------------
def secant_method(func, x0, x1, tol=1e-6, max_iter=100):

    for i in range(max_iter):

        f0 = func(x0)
        f1 = func(x1)

        if abs(f1 - f0) < 1e-12:
            break

        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)

        if abs(x2 - x1) < tol:
            return x2

        x0, x1 = x1, x2

    return x1


# ---------------------------------------------------------
# Main Program
# ---------------------------------------------------------
def main():

    print("Normal Distribution Probability Calculator")

    mu = float(input("Enter mean (mu): "))
    sigma = float(input("Enter standard deviation (sigma): "))
    mode = input("Are you specifying c or P? (enter 'c' or 'P'): ").strip()

    n = 1000

    if mode.lower() == 'c':

        c = float(input("Enter value of c: "))

        case = input(
            "Select case:\n"
            "1: P(mu-(c-mu) < x < mu+(c-mu))\n"
            "2: P(outside symmetric bounds)\n"
            "3: P(x < c)\n"
            "4: P(x > c)\n"
            "Enter 1,2,3,4: "
        )

        if case == '1':

            lower = mu - (c - mu)
            upper = mu + (c - mu)
            P = simpson(normal_pdf, lower, upper, n, mu, sigma)

        elif case == '2':

            lower = mu - (c - mu)
            upper = mu + (c - mu)

            P_inside = simpson(normal_pdf, lower, upper, n, mu, sigma)
            P = 1 - P_inside

        elif case == '3':

            P = simpson(normal_pdf, mu - 5 * sigma, c, n, mu, sigma)

        elif case == '4':

            P = 1 - simpson(normal_pdf, mu - 5 * sigma, c, n, mu, sigma)

        print(f"Probability = {P:.6f}")

    elif mode.lower() == 'P':

        target_P = float(input("Enter desired probability: "))

        case = input(
            "Select case:\n"
            "1: symmetric inside\n"
            "2: symmetric outside\n"
            "3: P(x < c)\n"
            "4: P(x > c)\n"
            "Enter 1,2,3,4: "
        )

        def equation(c):

            if case == '1':

                lower = mu - (c - mu)
                upper = mu + (c - mu)

                return simpson(normal_pdf, lower, upper, n, mu, sigma) - target_P

            elif case == '2':

                lower = mu - (c - mu)
                upper = mu + (c - mu)

                P_inside = simpson(normal_pdf, lower, upper, n, mu, sigma)

                return (1 - P_inside) - target_P

            elif case == '3':

                return simpson(normal_pdf, mu - 5 * sigma, c, n, mu, sigma) - target_P

            elif case == '4':

                return (1 - simpson(normal_pdf, mu - 5 * sigma, c, n, mu, sigma)) - target_P

        c_solution = secant_method(equation, mu, mu + sigma)

        print(f"Computed value of c = {c_solution:.6f}")

    else:
        print("Invalid selection.")


if __name__ == "__main__":
    main()
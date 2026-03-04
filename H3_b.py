"""
hw3b.py
MAE 3403 – HW 3

Computes the cumulative distribution function (CDF)
of the t-distribution:

F(z) = Km ∫_{-∞}^{z} (1 + u^2/m)^(-(m+1)/2) du

Km = Γ((m/2) + 1/2) / [sqrt(m*pi) * Γ(m/2)]

Gamma function computed numerically using Simpson’s rule.

Only uses: math
"""

import math


# ---------------------------------------------------------
# Gamma Function (Numerical Approximation)
# Γ(alpha) = ∫₀^∞ e^(-t) t^(alpha-1) dt
# ---------------------------------------------------------
def gamma_function(alpha):

    # upper limit large enough for convergence
    a = 0.0
    b = 50.0
    n = 5000

    if n % 2 == 1:
        n += 1

    h = (b - a) / n

    def integrand(t):
        return math.exp(-t) * (t ** (alpha - 1))

    s = integrand(a + 1e-10) + integrand(b)

    for i in range(1, n):
        x = a + i * h

        if i % 2 == 0:
            s += 2 * integrand(x)
        else:
            s += 4 * integrand(x)

    return (h / 3) * s


# ---------------------------------------------------------
# t-distribution integrand
# ---------------------------------------------------------
def t_integrand(u, m):
    return (1 + (u ** 2) / m) ** (-(m + 1) / 2)


# ---------------------------------------------------------
# Simpson’s rule for t-distribution integral
# ---------------------------------------------------------
def simpson_t(a, b, n, m):

    if n % 2 == 1:
        n += 1

    h = (b - a) / n
    s = t_integrand(a, m) + t_integrand(b, m)

    for i in range(1, n):

        x = a + i * h

        if i % 2 == 0:
            s += 2 * t_integrand(x, m)
        else:
            s += 4 * t_integrand(x, m)

    return (h / 3) * s


# ---------------------------------------------------------
# t-distribution CDF
# ---------------------------------------------------------
def t_cdf(z, m):

    # Compute normalization constant Km
    gamma1 = gamma_function((m / 2) + 0.5)
    gamma2 = gamma_function(m / 2)

    Km = gamma1 / (math.sqrt(m * math.pi) * gamma2)

    # approximate -∞ with large negative number
    lower = -50.0
    upper = z

    n = 2000

    integral = simpson_t(lower, upper, n, m)

    return Km * integral


# ---------------------------------------------------------
# Main Program
# ---------------------------------------------------------
def main():

    print("t-Distribution CDF Calculator")

    m = int(input("Enter degrees of freedom (m): "))
    z = float(input("Enter value of z: "))

    probability = t_cdf(z, m)

    print(f"\nF({z}) with m = {m} degrees of freedom")
    print(f"Probability = {probability:.6f}")


if __name__ == "__main__":
    main()
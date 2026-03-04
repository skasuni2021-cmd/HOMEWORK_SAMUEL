"""
Problem 1 (SciPy version) – Corrected
Simulate an industrial-scale gravel production process.
Rocks are assumed spherical and follow a log-normal size distribution before sieving.
After sieving between two screens (apertures Dmin and Dmax), the distribution is truncated.
The program prompts the user for µ (mean of ln(D)), σ (stdev of ln(D)), Dmax and Dmin,
then generates 11 samples of 100 rocks each from the truncated distribution.
For each sample the mean (D̅) and variance (S²) are printed, followed by the mean and
variance of the 11 sample means.
Numerical integration is done with scipy.integrate.quad, root finding with scipy.optimize.fsolve.
"""

import math
from random import random as rnd
import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
from matplotlib import pyplot as plt   # optional, for plotting

# ----------------------------------------------------------------------
# Probability density functions
# ----------------------------------------------------------------------
def ln_PDF(x, mu, sigma):
    """Standard log‑normal PDF."""
    if x <= 0.0:
        return 0.0
    return (1.0 / (x * sigma * math.sqrt(2.0 * math.pi))) * \
           math.exp(-((math.log(x) - mu) ** 2) / (2.0 * sigma ** 2))

def tln_PDF(x, mu, sigma, F_DMin, F_DMax):
    """Truncated log‑normal PDF (normalized between Dmin and Dmax)."""
    if x <= 0:
        return 0.0
    return ln_PDF(x, mu, sigma) / (F_DMax - F_DMin)

def F_tlnpdf(x, mu, sigma, D_Min, D_Max, F_DMin, F_DMax):
    """
    Cumulative distribution function of the truncated log‑normal from D_Min to x.
    Converts x to float to handle array input from fsolve.
    """
    x = float(x)                         # <-- crucial fix
    if x <= D_Min:
        return 0.0
    if x >= D_Max:
        return 1.0
    integral, _ = quad(lambda t: tln_PDF(t, mu, sigma, F_DMin, F_DMax), D_Min, x)
    return integral

# ----------------------------------------------------------------------
# Helper functions for user input and sample statistics
# ----------------------------------------------------------------------
def getPreSievedParameters(defaults):
    mu_def, sigma_def = defaults
    mu_str = input(f'Mean of ln(D) for the pre‑sieved rocks? (ln({math.exp(mu_def):.1f}) = {mu_def:.3f}, D in inches): ').strip()
    mu = mu_def if mu_str == '' else float(mu_str)
    sigma_str = input(f'Standard deviation of ln(D) for the pre‑sieved rocks? ({sigma_def:.3f}): ').strip()
    sigma = sigma_def if sigma_str == '' else float(sigma_str)
    return mu, sigma

def getSieveParameters(defaults):
    Dmin_def, Dmax_def = defaults
    Dmax_str = input(f'Large aperture size? ({Dmax_def:.3f}): ').strip()
    Dmax = Dmax_def if Dmax_str == '' else float(Dmax_str)
    Dmin_str = input(f'Small aperture size? ({Dmin_def:.3f}): ').strip()
    Dmin = Dmin_def if Dmin_str == '' else float(Dmin_str)
    return Dmin, Dmax

def getSampleParameters(defaults):
    nSamples_def, nSize_def = defaults
    nSamples_str = input(f'How many samples? ({nSamples_def}): ').strip()
    nSamples = nSamples_def if nSamples_str == '' else int(nSamples_str)
    nSize_str = input(f'How many items in each sample? ({nSize_def}): ').strip()
    nSize = nSize_def if nSize_str == '' else int(nSize_str)
    return nSamples, nSize

def sampleStats(data, doPrint=False):
    n = len(data)
    mean = sum(data) / n
    var = sum((x - mean) ** 2 for x in data) / (n - 1)
    if doPrint:
        print(f"mean = {mean:.3f}, var = {var:.3f}")
    return mean, var

def getFDMaxFDMin(mu, sigma, D_Min, D_Max):
    F_DMin, _ = quad(lambda x: ln_PDF(x, mu, sigma), 0, D_Min)
    F_DMax, _ = quad(lambda x: ln_PDF(x, mu, sigma), 0, D_Max)
    return F_DMin, F_DMax

def makeSample(mu, sigma, D_Min, D_Max, F_DMin, F_DMax, N=100):
    probs = [rnd() for _ in range(N)]
    sample = []
    for p in probs:
        # Solve F_tlnpdf(D) = p for D
        x0 = (D_Min + D_Max) / 2.0
        root = fsolve(lambda d: F_tlnpdf(d, mu, sigma, D_Min, D_Max, F_DMin, F_DMax) - p,
                      x0)[0]
        # Clamp to interval (fsolve may give tiny numerical excursions)
        root = max(D_Min, min(D_Max, root))
        sample.append(root)
    return sample

def makeSamples(mu, sigma, D_Min, D_Max, F_DMin, F_DMax, nSize, nSamples, doPrint):
    samples = []
    means = []
    for i in range(nSamples):
        samp = makeSample(mu, sigma, D_Min, D_Max, F_DMin, F_DMax, nSize)
        samples.append(samp)
        m, v = sampleStats(samp, doPrint=False)
        means.append(m)
        if doPrint:
            print(f"Sample {i}: mean = {m:.3f}, var = {v:.3f}")
    return samples, means

# ----------------------------------------------------------------------
# Main program
# ----------------------------------------------------------------------
def main():
    # Default values (inches)
    mu_def = math.log(2.0)      # µ = ln(2) ≈ 0.693
    sigma_def = 1.0
    D_Min_def = 3.0 / 8.0       # 0.375 inches
    D_Max_def = 1.0
    nSamples_def = 11
    nSize_def = 100

    goAgain = True
    while goAgain:
        mu, sigma = getPreSievedParameters((mu_def, sigma_def))
        D_Min, D_Max = getSieveParameters((D_Min_def, D_Max_def))
        nSamples, nSize = getSampleParameters((nSamples_def, nSize_def))

        F_DMin, F_DMax = getFDMaxFDMin(mu, sigma, D_Min, D_Max)

        # Optional plotting (uncomment if desired)
        # x_full = [i * 0.1 for i in range(100)]
        # y_full = [ln_PDF(x, mu, sigma) for x in x_full]
        # x_trunc = [D_Min + i * (D_Max - D_Min) / 99 for i in range(100)]
        # y_trunc = [tln_PDF(x, mu, sigma, F_DMin, F_DMax) for x in x_trunc]
        # fig, ax1 = plt.subplots()
        # ax1.plot(x_full, y_full, 'b-', label='original log‑normal')
        # ax1.set_xlabel('D (inches)')
        # ax1.set_ylabel('f(D)', color='b')
        # ax2 = ax1.twinx()
        # ax2.plot(x_trunc, y_trunc, 'r-', label='truncated')
        # ax2.set_ylabel('f_trunc(D)', color='r')
        # plt.title('Original vs. truncated log‑normal PDF')
        # plt.show()

        print("\nGenerating samples...")
        samples, means = makeSamples(mu, sigma, D_Min, D_Max, F_DMin, F_DMax,

                                     mean_of_means, var_of_means=sampleStats(means)
        print(f"\nMean of the sampling mean: {mean_of_means:.3f}")
        print(f"Variance of the sampling mean: {var_of_means:.6f}")

        again = input('\nGo again? (No): ').strip().lower()
        goAgain = 'y' in again

        if __name__ == '__main__':
            main()
        nSize, nSamples, doPrint=True)

import numpy as np
import math
from scipy.stats import chi2

# Input data
data = list(map(int, input("Enter space-separated frequency values: ").split()))

# Basic statistics
N = len(data)
max_val = max(data)

# Frequency distribution (Observed)
observed_freq = [data.count(i) for i in range(max_val + 1)]
X = list(range(max_val + 1))  # X values
total_freq = sum(observed_freq)

# Probability for each X based on observed data
prob_obs = [f / total_freq for f in observed_freq]

# Compute mean (λ for Poisson) using dot product
mean = np.inner(X, prob_obs)

# Compute expected probabilities & expected frequencies under Poisson
expected_probs = []
expected_freq = []
chi_sq_components = []

print("\nX\tP(X=x)\tObs.Freq\tExp.Freq\tChi^2")
print("-" * 50)

for x in X:
    # Poisson probability formula: P(X = x) = e^(-λ) * λ^x / x!
    poisson_prob = math.exp(-mean) * (mean ** x) / math.factorial(x)
    exp_freq = poisson_prob * total_freq
    chi_sq = ((observed_freq[x] - exp_freq) ** 2) / exp_freq if exp_freq > 0 else 0

    expected_probs.append(poisson_prob)
    expected_freq.append(exp_freq)
    chi_sq_components.append(chi_sq)

    print(f"{x}\t{poisson_prob:.4f}\t{observed_freq[x]:>8}\t{exp_freq:>9.2f}\t{chi_sq:>7.2f}")

# Calculate total Chi-square value
calculated_chi_sq = sum(chi_sq_components)
degrees_of_freedom = max_val  # df = k - 1 - 1 (mean estimated → one parameter)
table_chi_sq = chi2.ppf(1 - 0.01, df=degrees_of_freedom)

# Hypothesis Testing
print("-" * 50)
print(f"Calculated Chi-square value: {calculated_chi_sq:.4f}")
print(f"Critical Chi-square value (1% LOS, df={degrees_of_freedom}): {table_chi_sq:.4f}")

if calculated_chi_sq < table_chi_sq:
    print("The data *fits* the Poisson distribution at 1% level of significance.")
else:
    print("The data *does not fit* the Poisson distribution at 1% level of significance.")
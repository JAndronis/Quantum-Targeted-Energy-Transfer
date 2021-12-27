import numpy as np

def construct_Hamiltonians(chiA, chiD, coupling_lambda, omegaA, omegaD, max_N):
  H = np.zeros((max_N + 1, max_N + 1), dtype=float)
  
  # i bosons at the donor
  for i in range(max_N + 1):
    for j in range(max_N + 1):
      # First term from interaction
      if i == j - 1: H[i][j] = -coupling_lambda * np.sqrt((i + 1) * (max_N - i))
      # Second term from interaction
      if i == j + 1: H[i][j] = -coupling_lambda * np.sqrt(i * (max_N - i + 1))
        # Term coming from the two independent Hamiltonians
      if i == j: H[i][j] = omegaD * i + 0.5 * chiD * i ** 2 + omegaA * (max_N - i) + 0.5 * chiA * (max_N - i) ** 2

  return H
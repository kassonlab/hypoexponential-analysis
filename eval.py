# Code 2024-2025 by Oyin Adenekan
# edited by Peter Kasson
"""Model evaluation routines"""

import numpy as np
import scipy

def calculate_aic(likelihoods, num_parameters):
  return -2*np.max(likelihoods) + 2*num_parameters

def calculate_dic(likelihoods):
  # after_burnin_likelihood = likelihoods[100:]
  return np.max(likelihoods) - 2*np.var(likelihoods[likelihoods > -np.inf])

def calculate_aicc(likelihoods, num_parameters, sample_size):
  # PK correcting
  # AICC = -2 * logL + 2*k + 2k^2+2k/(n-k-1) where k = parameters, n = sample size
  return -2*np.max(likelihoods) + 2*num_parameters + 2 * (num_parameters**2 + num_parameters) / (sample_size - num_parameters - 1)
  # old: return -2*np.max(likelihoods) + 2*num_parameters*(sample_size/(num_parameters - sample_size - 1))

def calculate_bic(likelihoods, num_parameters, sample_size):
  return -2*np.max(likelihoods) + num_parameters*np.log(sample_size)

def calculate_randomness_parameter(dwell_times):
  first_term = np.mean(dwell_times**2)
  second_term = np.mean(dwell_times)**2
  return (first_term - second_term) / second_term

def calc_aicc(likelihoods2, likelihoods3, likelihoods4, likelihoods5, data,
              burnin=1000):
  # find optimal number for N, aicc
  data_ecdf = scipy.stats.ecdf(data)
  num_times = data_ecdf.cdf.quantiles.shape[0]
  aicc_1 = 1e99  # skip
  aicc_2 = calculate_aicc(likelihoods2[burnin:], 2, num_times)
  aicc_3 = calculate_aicc(likelihoods3[burnin:], 3, num_times)
  aicc_4 = calculate_aicc(likelihoods4[burnin:], 4, num_times)
  aicc_5 = calculate_aicc(likelihoods5[burnin:], 5, num_times)
  aiccs = np.array([aicc_1, aicc_2, aicc_3, aicc_4, aicc_5])
  optimal_num_steps_aicc_content = np.argmin(aiccs) + 1
  print(aiccs)
  print(optimal_num_steps_aicc_content)
  return aiccs

def calc_bic(likelihoods2, likelihoods3, likelihoods4, likelihoods5, data,
             burnin=1000):
  # find optimal number for N, bic
  data_ecdf = scipy.stats.ecdf(data)
  num_times = data_ecdf.cdf.quantiles.shape[0]
  bic_1 = 1e99  # skip
  bic_2 = calculate_bic(likelihoods2[burnin:], 2, num_times)
  bic_3 = calculate_bic(likelihoods3[burnin:], 3, num_times)
  bic_4 = calculate_bic(likelihoods4[burnin:], 4, num_times)
  bic_5 = calculate_bic(likelihoods5[burnin:], 5, num_times)
  bics = np.array([bic_1,bic_2, bic_3, bic_4, bic_5])
  optimal_num_steps_bic_content = np.argmin(bics) + 1
  print(bics)
  print(optimal_num_steps_bic_content)
  return bics

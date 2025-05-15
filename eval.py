# Code 2024-2025 by Oyin Adenekan
# edited by Peter Kasson
# Model evaluation routines

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

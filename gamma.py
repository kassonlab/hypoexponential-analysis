# Code 2024-2025 by Oyin Adenekan
# edited by Peter Kasson
""" Routines for gamma fits."""

# gamma pdf
def gamma(k, N, x):
  return ((k**N)*(x**(N-1)))*np.exp(-k*x)/scipy.special.factorial(N) # formula for convolution of N exponentials, 1 k for each step

def simulate_gamma(k, N, num_times=300, upper_time_bound=500):
  # set things up
  lower_time_bound = 0
  dwell_times = np.zeros((num_times))
  prob_dwell_times = np.zeros((num_times))
  successes = []

  # perform simulations
  total_iters = 0
  idx = 0
  while idx < num_times:
    # R = np.random.rand(lower_time_bound, upper_time_bound) # propose a dwell time in the range of ks
    R = np.random.random_sample()*upper_time_bound # propose a dwell time in the range of ks
    prob_R = gamma(k, N, R) # get probability of dwell time according to pdf
    # print(R, prob_R)
    prob_accept = np.random.rand()
    if prob_R > prob_accept:
      # print(idx, 'success !')
      dwell_times[idx] = R
      prob_dwell_times[idx] = prob_R
      idx = idx + 1
      successes.append(True)
    else:
      successes.append(False)
    # total_iters = total_iters + 1

  return dwell_times

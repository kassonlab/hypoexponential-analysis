
# Original code 2024-2025 by Oyin Adenekan
# edited by Peter Kasson
# functions for MCMC sampling of hypoexponential distributions

import numpy as np
import scipy

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

# calculate hyperexponential pdf
def conv_1_exps(rates, t):
  return rates[0]*np.exp(-rates[0]*t)
def conv_2_exps(rates, t):
  if rates[0] == rates[1]:
    return rates[0]*rates[1]*t*np.exp(-rates[0]*t)
  else:
    return ((rates[0]*rates[1])/(rates[1]-rates[0]))*(np.exp(-rates[0]*t)-np.exp(-rates[1]*t))
def conv_3_exps(rates, t):
  first_term = (rates[0]*rates[1]*rates[2])/(rates[1]-rates[0])
  second_term_1  = (np.exp(-rates[0]*t)-np.exp(-rates[2]*t))/(-rates[0]+rates[2])
  second_term_2 =  (np.exp(-rates[1]*t)-np.exp(-rates[2]*t))/(-rates[1]+rates[2])
  return first_term*(second_term_1 - second_term_2)
def conv_4_exps(rates, t):
  k1 = rates[0]
  k2 = rates[1]
  k3 = rates[2]
  k4 = rates[3]
  pre_num = k1*k2*k3*k4
  mid_term = (np.exp(-k1*t) - np.exp(-k3*t))/(k3 - k1) - (np.exp(-k2*t) - np.exp(-k3*t))/(k3 - k2)
  k4_term = 1/k4 - np.exp(-k4*t)/k4
  big_denom = k2-k1
  return (pre_num*mid_term*k4_term)/big_denom
def conv_5_exps(rates, t):
  k1 = rates[0]
  k2 = rates[1]
  k3 = rates[2]
  k4 = rates[3]
  k5 = rates[4]
  pre_term = k1*k2*k3*k4*k5
  first_mid = (np.exp(-k1*t) - np.exp(-k3*t))/(k3 - k1)
  second_mid = (np.exp(-k2*t) - np.exp(-k3*t))/(k3 - k2)
  k4_term = 1/k4 - np.exp(-k4*t)/k4
  k5_term = 1/k5 - np.exp(-k5*t)/k5
  big_denom = k2-k1
  return pre_term* (first_mid - second_mid) * k4_term * k5_term / big_denom

# simulate dwell times for a given set of rates
def simulate_conv_exps(convolution_func, rates, num_times=300, upper_time_bound=3000):
# set things up
  lower_time_bound = 0
  dwell_times = np.zeros((num_times))
  prob_dwell_times = np.zeros((num_times))
  successes = []

  total_iters = 0
  idx = 0
  while idx < num_times:
    # R = np.random.rand(lower_time_bound, upper_time_bound) # propose a dwell time in the range of ks
    R = np.random.random_sample()*upper_time_bound # propose a dwell time in the range of ks
    prob_R = convolution_func(rates, R) # get probability of dwell time according to pdf
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

# perform mcmc
def do_mcmc(num_iters, num_rates, datas, conv_func):

  # initialize values
  low_bound = 0
  high_bound = 1
  k_current = np.random.uniform(low=low_bound, high=high_bound, size=num_rates)
  temperature = 1


  # variables for for loop
  ks = np.zeros((num_rates, num_iters))

  # tracking accepts
  accepts = np.zeros((num_rates, num_iters))
  accepts[:, 0] = 1
  # accept_interval_size = 500
  # delta_std_dev = 0.3

  # tracking proposals
  likelihood_curve = np.zeros((num_iters))
  likelihood_curve[0] = np.sum(np.log(conv_func(k_current, datas)))
  ks[:,0] = k_current
  proposals = np.zeros((num_rates, num_iters))
  proposals[:, 0] = k_current



  # track accept probabilities
  accept_probs = np.zeros((num_rates, num_iters-1))

  print_flag = 1


  for iter in range(1, num_iters):
    if iter > 100:
      print_flag = 0
    if iter % 100 == 0:
      print_flag = 1;
    # if print_flag: print('iter', iter)


    for idx in range(num_rates):
      # if print_flag: print('which k: {}'.format(idx+1))

      # calculate the likelihoods, for new and old
      # if print_flag: print('current')
      # if print_flag: print('k1: {}, k2: {}, k3: {}'.format(k_current[0], k_current[1], k_current[2]))
      prev_likelihood = np.sum(np.log(conv_func(k_current, datas)))
      # if print_flag: print('current likelihood:', prev_likelihood)

      # propose a proposal
      k_proposed = np.random.gamma(shape=1, scale=k_current[idx])
      proposals[idx, iter] = k_proposed

      # update current list and keep pre proposal number
      pre_proposal_rate = k_current[idx]
      k_current[idx] = k_proposed
      # k_proposed = np.random.gamma(this_k, 10)


      # if print_flag: print('proposed')
      # if print_flag: print('k1: {}, k2: {}, k3: {}'.format(k_current[0], k_current[1], k_current[2]))
      new_likelihood = np.sum(np.log(conv_func(k_current, datas)))
      # if print_flag: print('proposed likelihood:', new_likelihood)

      accept_ratio = np.exp(new_likelihood - prev_likelihood)
      # accept_ratio = new_likelihood/prev_likelihood
      # if print_flag: print('accept ratio: {}'.format(accept_ratio))
      prob_accept = accept_ratio if accept_ratio < 1 else 1
      accept_probs[idx, iter-1] = accept_ratio
      # if print_flag: print('calculated accept prob: {}, actual accept prob: {}'.format(accept_ratio, prob_accept))
      if accept_ratio**temperature >= np.random.uniform():
        ks[idx, iter] = k_proposed
        likelihood_curve[iter] = new_likelihood
        # k_current[idx] = k_proposed
        accepts[idx, iter] = 1
      else:
        k_current[idx] = pre_proposal_rate
        ks[idx, iter] = k_current[idx]
        likelihood_curve[iter] = prev_likelihood

  return ks, likelihood_curve

# dwell times to ecdf
def dwell_times_to_ecdf(dwell_times):
  num_dwell_times = len(dwell_times)
  dwell_times_sorted_args = np.argsort(dwell_times)
  dwell_times_sorted = dwell_times[dwell_times_sorted_args]
  # time_points = np.linspace(0, np.max(dwell_times), num_)
  ecdf = np.zeros((num_dwell_times))
  for idx in range(num_dwell_times):
    num_fused = len(dwell_times_sorted[dwell_times_sorted <= dwell_times_sorted[idx]])
    ecdf[idx] = num_fused/num_dwell_times
  return ecdf

# dwell times, w independent max value, to ecdf
def dwell_times_to_ecdf_independent_max(dwell_times, max, length):
  # create an x-axis based on given max time
  x_axis = np.linspace(0, max, length)
  ecdf = np.zeros((length))
  for idx in range(len(x_axis)):
    num_dwell_times_in_interval = len(dwell_times[dwell_times <= x_axis[idx]])
    ecdf[idx] = num_dwell_times_in_interval/length
  return x_axis, ecdf

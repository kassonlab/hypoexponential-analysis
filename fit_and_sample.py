# Code 2024-2025 by Oyin Adenekan
# edited by Peter Kasson
""" Routines to apply hypoexponential fits, select best model, sample from process to compare dwell times."""

import numpy as np
import scipy

from . import core
from . import eval
from . import fits
from . import gillespie

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
def dwell_times_to_ecdf_independent_max(dwell_times, max_time, length):
  # create an x-axis based on given max time
  x_axis = np.linspace(0, max_time, length)
  ecdf = np.zeros((length))
  for idx in range(len(x_axis)):
    num_dwell_times_in_interval = len(dwell_times[dwell_times <= x_axis[idx]])
    ecdf[idx] = num_dwell_times_in_interval/length
  return x_axis, ecdf

# make a function to create information for ecdf
def make_ecdf_inputs(data, return_all = False, mcmc_steps=10000, burnin=100):
  # make ecdf for actual data
  data_ecdf = scipy.stats.ecdf(data)
  num_times = data_ecdf.cdf.quantiles.shape[0]
  # upper_time_bound = data_ecdf.cdf.quantiles[-1]


  # fit and simulate for 1
  ks_1, likelihoods_1 = fits.do_mcmc(mcmc_steps, 1, data, core.conv_1_exps)
  max_likelihood_idx_1 = np.argmax(likelihoods_1)
  ks_max_likelihood_1 = ks_1[:, max_likelihood_idx_1]

  # fit and simulate for 2
  ks_2, likelihoods_2 = fits.do_mcmc(mcmc_steps, 2, data, core.conv_2_exps)
  max_likelihood_idx_2 = np.argmax(likelihoods_2)
  ks_max_likelihood_2 = ks_2[:, max_likelihood_idx_2]

  # fit and simulate for 3
  ks_3, likelihoods_3 = fits.do_mcmc(mcmc_steps, 3, data, core.conv_3_exps)
  max_likelihood_idx_3 = np.argmax(likelihoods_3)
  ks_max_likelihood_3 = ks_3[:, max_likelihood_idx_3]

  # fit and simulate for 4
  ks_4, likelihoods_4 = fits.do_mcmc(mcmc_steps, 4, data, core.conv_4_exps)
  max_likelihood_idx_4 = np.argmax(likelihoods_4)
  ks_max_likelihood_4 = ks_4[:, max_likelihood_idx_4]

  # fit and simulate for 5
  ks_5, likelihoods_5 = fits.do_mcmc(mcmc_steps, 5, data, core.conv_5_exps)
  max_likelihood_idx_5 = np.argmax(likelihoods_5)
  ks_max_likelihood_5 = ks_5[:, max_likelihood_idx_5]


  # find optimal number for N, aicc
  aicc_1 = eval.calculate_aicc(likelihoods_1[burnin:], 1, num_times)
  aicc_2 = eval.calculate_aicc(likelihoods_2[burnin:], 2, num_times)
  aicc_3 = eval.calculate_aicc(likelihoods_3[burnin:], 3, num_times)
  aicc_4 = eval.calculate_aicc(likelihoods_4[burnin:], 4, num_times)
  aicc_5 = eval.calculate_aicc(likelihoods_5[burnin:], 5, num_times)

  aiccs = np.array([aicc_1, aicc_2, aicc_3, aicc_4, aicc_5])
  optimal_num_steps_aicc_content = np.argmin(aiccs) + 1
  print(optimal_num_steps_aicc_content)

  # simulate data under hypoexponenetial

  sim_data = None
  ks_max_likelihood_optimal = None
  if optimal_num_steps_aicc_content == 1:
    ks_max_likelihood_optimal = ks_max_likelihood_1
    min_rate = np.min(ks_max_likelihood_optimal)
    upper_time_bound = int(1/min_rate*10*1.5)
    sim_data = gillespie.sim_gillespie_one_step(ks_max_likelihood_optimal, num_times, upper_time_bound)
  elif optimal_num_steps_aicc_content == 2:
    ks_max_likelihood_optimal = ks_max_likelihood_2
    min_rate = np.min(ks_max_likelihood_optimal)
    upper_time_bound = int(1/min_rate*10*1.5)
    sim_data = gillespie.sim_gillespie_two_step(ks_max_likelihood_optimal, num_times, upper_time_bound)
  elif optimal_num_steps_aicc_content == 3:
    ks_max_likelihood_optimal = ks_max_likelihood_3
    min_rate = np.min(ks_max_likelihood_optimal)
    upper_time_bound = int(1/min_rate*10*1.5)
    sim_data = gillespie.sim_gillespie_three_step(ks_max_likelihood_optimal, num_times, upper_time_bound)
  elif optimal_num_steps_aicc_content == 4:
    ks_max_likelihood_optimal = ks_max_likelihood_4
    min_rate = np.min(ks_max_likelihood_optimal)
    upper_time_bound = int(1/min_rate*10*1.5)
    sim_data = gillespie.sim_gillespie_four_step(ks_max_likelihood_optimal, num_times, upper_time_bound)
  elif optimal_num_steps_aicc_content == 5:
    ks_max_likelihood_optimal = ks_max_likelihood_5
    min_rate = np.min(ks_max_likelihood_optimal)
    upper_time_bound = int(1/min_rate*10*1.5)
    sim_data = gillespie.sim_gillespie_five_step(ks_max_likelihood_optimal, num_times, upper_time_bound)

  if return_all:
    kmax_all = (ks_max_likelihood_1, ks_max_likelihood_2, ks_max_likelihood_3, ks_max_likelihood_4, ks_max_likelihood_5)
    ks_all = (ks_1[:, burnin:], ks_2[:, burnin:], ks_3[:, burnin:], ks_4[:, burnin:], ks_5[:, burnin:])
    likelihoods_all = [likelihoods_1[burnin:], likelihoods_2[burnin:],
                       likelihoods_3[burnin:], likelihoods_4[burnin:], likelihoods_5[burnin:]]
    return {'best_k': ks_max_likelihood_optimal, 'sim_data' : sim_data, 'all_ks' : ks_all,
            'all_likelihood': likelihoods_all, 'each_best_k' : kmax_all, 'aicc' : aiccs}
  else:
    return ks_max_likelihood_optimal, sim_data

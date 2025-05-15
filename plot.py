# Code 2024-2025 by Oyin Adenekan
# edited by Peter Kasson
# Plots for MCMC hypoexponential fitting 

import numpy as np
import matplotlib.pyplot as plt

# plotting the mcmc fit
def ks_likelihood_plot(ks, likelihoods, rates, sim_data):

  # determine which ks are which ks -- sort each iteration of mcmc from least to greatest
  sorted_ks = np.sort(ks, axis=0)

  # getting likelihoods and ks after burn in iterations
  burnin = 100
  k1s = sorted_ks[0, burnin:]
  k2s = sorted_ks[1, burnin:]
  relevant_likelihoods = likelihoods[burnin:]
  max_likelihood_arg = np.argmax(relevant_likelihoods)

  # get ground truths
  ground_truth_k1, ground_truth_k2 = np.sort(rates)
  ground_truth_likelihood = np.sum(np.log(conv_2_exps(rates, sim_data)))

  # print the rates w maximum likelihood
  # print('k1: {}, k2: {}'.format(k1s[max_likelihood_arg], k2s[max_likelihood_arg]))

  # plotting a figure
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')
  ax.scatter(k1s, k2s, relevant_likelihoods, marker='.', c='grey')
  ax.scatter(ground_truth_k1, ground_truth_k2, ground_truth_likelihood, marker='*', s = 200, c='red', label='ground truth')
  ax.scatter(k1s[max_likelihood_arg], k2s[max_likelihood_arg], relevant_likelihoods[max_likelihood_arg], marker='o', s = 200, c='red', label='maximum likelihood')
  ax.set_xlabel('k1')
  ax.set_ylabel('k2')
  ax.set_zlabel('log likelihood')
  ax.set_title('{}, {}'.format(ground_truth_k1, ground_truth_k2))
  plt.title(title)
  plt.legend()

# simulate plot of ecdfs of different steps compared to one another
def simulate_result_plot_ecdf(actual_data, rates_1_step, rates_2_step, rates_3_step, num_times, upper_time_bound):

  # lets get a max
  max_time = np.max(np.concatenate((actual_data, rates_1_step, rates_2_step, rates_3_step), axis=None))

  num_time_intervals = 500
  # simulate data for 1, 2, 3 steps
  # actual_ecdf = dwell_times_to_ecdf(actual_data)
  actual_x, actual_ecdf = dwell_times_to_ecdf_independent_max(actual_data, max_time, num_time_intervals)

  simulated_data_1 = simulate_conv_exps(conv_1_exps, rates_1_step, num_times=300, upper_time_bound=upper_time_bound)
  # simulated_data_ecdf_1 = dwell_times_to_ecdf(simulated_data_1)
  x_1, simulated_data_ecdf_1 = dwell_times_to_ecdf_independent_max(simulated_data_1, max_time, num_time_intervals)



  simulated_data_2 = simulate_conv_exps(conv_2_exps, rates_2_step, num_times=300, upper_time_bound=upper_time_bound)
  # simulated_data_ecdf_2 = dwell_times_to_ecdf(simulated_data_2)
  x_2, simulated_data_ecdf_2 = dwell_times_to_ecdf_independent_max(simulated_data_2, max_time, num_time_intervals)


  simulated_data_3 = simulate_conv_exps(conv_3_exps, rates_3_step, num_times=300, upper_time_bound=upper_time_bound)
  # simulated_data_ecdf_3 = dwell_times_to_ecdf(simulated_data_3)
  x_3, simulated_data_ecdf_3 = dwell_times_to_ecdf_independent_max(simulated_data_3, max_time, num_time_intervals)


  # plt.figure()
  # plt.plot(np.sort(actual_data), actual_ecdf, '.', label='actual', markersize = 3)
  # plt.plot(np.sort(simulated_data_1), simulated_data_ecdf_1, '.', label='1 step', markersize = 3)
  # plt.plot(np.sort(simulated_data_2), simulated_data_ecdf_2, '.', label='2 steps', markersize = 3)
  # plt.plot(np.sort(simulated_data_3), simulated_data_ecdf_3, '.', label='3 steps', markersize = 3)
  # plt.legend()
  # plt.xlabel('time (s)')
  # plt.ylabel('eCDF')

  plt.figure()
  plt.plot(actual_x, actual_ecdf, '.', label='actual', markersize = 3)
  plt.plot(x_1, simulated_data_ecdf_1, '.', label='1 step', markersize = 3)
  plt.plot(x_2, simulated_data_ecdf_2, '.', label='2 steps', markersize = 3)
  plt.plot(x_3, simulated_data_ecdf_3, '.', label='3 steps', markersize = 3)
  plt.legend()
  plt.xlabel('time (s)')
  plt.ylabel('eCDF')


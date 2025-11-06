# Code 2025 by Peter Kasson

import numpy as np
import scipy
import core
import eval
import fits

def multi_path_likelihood(Gvals, Temp, times):
  """Compute multi-path likelihood.
  Args:
    Gvals: [dHact_prot, dH_prot, dHact_fuse, dSact_prot, dS_prot, dSact_fuse]
    Temp
    times:  list of waiting times.
  """

  # trying units of kT for everything
  dGact_prot = Gvals[0] + Temp/298*Gvals[3]
  dG_prot = Gvals[1] + Temp/298*Gvals[4]
  dGact_fuse = Gvals[2] + Temp/298*Gvals[5]
  # kb/h = 2.084e10 s-1
  prefactor = 2.084e10 * Temp
  kT = 1.987e-3 * Temp  # kcal mol-1

  # set rates
  # should differentiate dGact_prot from dG_prot
  kprot = prefactor * np.exp(-dGact_prot*298/Temp)
  kfuse_1 = prefactor * np.exp((-dGact_fuse+dG_prot)*298/Temp)
  kfuse_2 = prefactor * np.exp((-dGact_fuse+2*dG_prot)*298/Temp)
  kfuse_3 = prefactor * np.exp((-dGact_fuse+3*dG_prot)*298/Temp)
  kfuse_4 = prefactor * np.exp((-dGact_fuse+4*dG_prot)*298/Temp)

  # compute and return likelihoods
  P_one = core.conv_2_exps([kprot, kfuse_1], times)
  P_two = core.conv_3_exps([kprot, kprot, kfuse_2], times)
  P_three = core.conv_4_exps([kprot, kprot, kprot, kfuse_3], times)
  P_four = core.conv_5_exps([kprot, kprot, kprot, kprot, kfuse_4], times)
  # technically should be max, but I think we can approximate as sum
  return np.max([np.nan_to_num(P_one), np.nan_to_num(P_two),
                 np.nan_to_num(P_three), np.nan_to_num(P_four)])


def multi_path_multi_temp_likelihood(Gvals, data):
  """Compute multi-path likelihood over a multi-temperature series.
  Args:
    Gvals: [dGact_prot, dGac_fuse]
    data:  {Temp1 : times1, Temp2 : times2, ...}
  """
  # Calculate log-likelihood for each temperature and sum them up
  total_log_likelihood = 0
  for Temp_C, times in data.items():
    # *** Convert Temp_C to Kelvin ***
    total_log_likelihood += np.sum(np.log(multi_path_likelihood(Gvals, Temp_C+273, times)))
  return total_log_likelihood


def make_ecdf_inputs(data, return_all = False, mcmc_steps=10000, burnin=100):
  """Determine fits, make ECDF."""
  data_ecdf = [scipy.stats.ecdf(data_series) for data_series in data.values()]
  num_times = [x.cdf.quantiles.shape[0] for x in data_ecdf]
  sim_data = None
  [Gvals, likelihoods] = fits.do_mcmc(mcmc_steps, 6, data, multi_path_multi_temp_likelihood, bounds=[0, 20])
  # Skipping data simulation for now, will need to add that
  max_likelihood_index = np.argmax(likelihoods)
  return {'best_G': Gvals[:, max_likelihood_index], 'sim_data' : sim_data, 'all_Gs' : Gvals,
          'all_likelihood': likelihoods, 'aicc' : eval.calculate_aicc(likelihoods[burnin:], 3, np.sum(num_times))}

# Alternatives for rate space sampling
def multi_path_likelihood_rs(kvals, Temp, times,
                             separate_entropy=True):
  """Compute multi-path likelihood.
  Args:
    kvals: [kact_prot, kact_fuse, k_prot]
    Temp:  temperature
    times:  list of waiting times.
  """
  kbh = 2.084e10  # s-1
  prefactor_std = kbh*298
  prefactor = 2.084e10 * Temp
  # need to temperature-correct the rates
  # kper_prot should be >1, other k's should be <1

  kprot_s = kvals[3] if separate_entropy else 1
  kfuse_s = kvals[4] if separate_entropy else 1
  kper_prot_s = kvals[5] if separate_entropy else 1

  """Old setup:
  kprot = prefactor * (kvals[0] / prefactor_std)**(298/Temp) * kprot_s
  kfuse = prefactor * (kvals[1] / prefactor_std)**(298/Temp) * kfuse_s
  kper_prot = kvals[2]**(298/Temp) * kper_prot_s
  """
  # New setup:
  kprot = Temp/298 * kvals[0]**(298/Temp) * kprot_s
  kfuse = Temp/298 * kvals[1]**(298/Temp) * kfuse_s
  kper_prot = kvals[2]**(298/Temp) * kper_prot_s
  # I think the problem may be this kb/h

  kfuse_1 = kfuse/kper_prot
  kfuse_2 = kfuse/(kper_prot*kper_prot)
  kfuse_3 = kfuse/(kper_prot*kper_prot*kper_prot)
  kfuse_4 = kfuse/(kper_prot*kper_prot*kper_prot*kper_prot)

  # kT = 1.987e-3 * Temp  # kcal mol-1
  # dGact_prot = np.log(kvals[0]/prefactor) * 298/Temp  # includes kT
  # kprot = prefactor * np.exp(-dGact_prot)
  # dG_prot = np.log(kvals[2]) * 298/Temp # includes kT
  # dGact_fuse = np.log(kvals[1]/prefactor) * 298/Temp  # includes kT
  # kprot = prefactor * np.exp(-dGact_prot)
  # kfuse_1 = prefactor * np.exp(-dGact_fuse) * np.exp(dG_prot)
  # kfuse_2 = prefactor * np.exp(-dGact_fuse) * np.exp(2*dG_prot)
  # kfuse_3 = prefactor * np.exp(-dGact_fuse) * np.exp(3*dG_prot)
  # kfuse_4 = prefactor * np.exp(-dGact_fuse) * np.exp(4*dG_prot)

  # kfuse_1 = prefactor * np.exp((-dGact_fuse+dG_prot)/kT) = prefactor * np.exp(-dGact_fuse/KT) * np.exp(dG_prot/kT)


  # compute and return likelihoods
  P_one = core.conv_2_exps([kprot, kfuse_1], times)
  P_two = core.conv_3_exps([kprot, kprot, kfuse_2], times)
  P_three = core.conv_4_exps([kprot, kprot, kprot, kfuse_3], times)
  P_four = core.conv_5_exps([kprot, kprot, kprot, kprot, kfuse_4], times)
  return (np.nan_to_num(P_one) + np.nan_to_num(P_two)
          + np.nan_to_num(P_three) + np.nan_to_num(P_four))

def likelihood_ratespace_perprot(kvals, Temp, times,
                                 separate_entropy=True, separate_protG=False):
  """Compute multi-path likelihood. Incorporates limits on per-protein rate.
  Args:
    kvals: [kact_prot, kact_fuse, k_prot]
    Temp:  temperature
    times:  list of waiting times.
  """
  kbh = 2.084e10  # s-1
  prefactor_std = kbh*298
  prefactor = 2.084e10 * Temp
  # need to temperature-correct the rates
  # kper_prot should be >1, other k's should be <1
  if separate_protG:
    kprot_s = kvals[3] if separate_entropy else 1
    kfuse_s = kvals[4] if separate_entropy else 1
    kper_prot_s = kvals[5] if separate_entropy else 1
  else:
    kprot_s = kvals[2] if separate_entropy else 1
    kfuse_s = kvals[3] if separate_entropy else 1

  """Old setup:
  kprot = prefactor * (kvals[0] / prefactor_std)**(298/Temp) * kprot_s
  kfuse = prefactor * (kvals[1] / prefactor_std)**(298/Temp) * kfuse_s
  kper_prot = kvals[2]**(298/Temp) * kper_prot_s
  """
  # New setup:
  kprot = Temp/298 * kvals[0]**(298/Temp) * kprot_s
  kfuse = Temp/298 * kvals[1]**(298/Temp) * kfuse_s
  if separate_protG:
    # idea is that ∆Gprot = ∆Gact_prot - ∆Gactrev_prot
    # and kfuse = kbT/h exp(-∆Gfuse/kbT +∆Gprot/kbT)
    # = kbT/h exp(-∆Gfuse)/exp(-∆Gprot/kbT)
    # = kbT/h exp(-∆Gfuse)/exp(-∆Gact_prot/kT)*exp(-∆Gact_revprot/kT)
    # so k_per_prot
    kper_prot = kprot/prefactor / (kvals[2]**(298/Temp) * kper_prot_s)
  else:
    kper_prot = kprot / prefactor
  # I think the problem may be this kb/h

  kfuse_1 = kfuse/kper_prot
  kfuse_2 = kfuse/(kper_prot*kper_prot)
  kfuse_3 = kfuse/(kper_prot*kper_prot*kper_prot)
  kfuse_4 = kfuse/(kper_prot*kper_prot*kper_prot*kper_prot)

  # compute and return likelihoods
  P_one = core.conv_2_exps([kprot, kfuse_1], times)
  P_two = core.conv_3_exps([kprot, kprot, kfuse_2], times)
  P_three = core.conv_4_exps([kprot, kprot, kprot, kfuse_3], times)
  P_four = core.conv_5_exps([kprot, kprot, kprot, kprot, kfuse_4], times)
  # again experimenting with max
  return np.max([np.nan_to_num(P_one), np.nan_to_num(P_two),
                 np.nan_to_num(P_three), np.nan_to_num(P_four)])

def multi_path_multi_temp_likelihood_rs(kvals, data):
  """Compute multi-path likelihood over a multi-temperature series.
  Args:
    kvals: [kact_prot, kact_fuse, k_prot]
    data:  {Temp1 : times1, Temp2 : times2, ...}
  """
  # Calculate log-likelihood for each temperature and sum them up
  total_log_likelihood = 0
  for Temp_C, times in data.items():
    # *** Convert temp to Kelvin ***
    total_log_likelihood += np.sum(np.log(multi_path_likelihood_rs(kvals, Temp_C+273, times)))
  return total_log_likelihood

def multi_temp_likelihood_perprot_rs(kvals, data):
  """Compute multi-path likelihood over a multi-temperature series.
  Args:
    kvals: [kact_prot, kact_fuse, k_prot]
    data:  {Temp1 : times1, Temp2 : times2, ...}
  """
  # Calculate log-likelihood for each temperature and sum them up
  total_log_likelihood = 0
  for Temp_C, times in data.items():
    # *** Convert temp to Kelvin ***
    total_log_likelihood += np.sum(np.log(likelihood_ratespace_perprot(kvals, Temp_C+273, times, separate_protG=True)))
  return total_log_likelihood

def make_ecdf_inputs_rs(data, return_all = False, mcmc_steps=10000, burnin=100,
                        separate_entropy=True,
                        likelihood_fn=multi_path_multi_temp_likelihood_rs):
  """Determine fits, make ECDF using rate-space sampling."""
  data_ecdf = [scipy.stats.ecdf(data_series) for data_series in data.values()]
  num_times = [x.cdf.quantiles.shape[0] for x in data_ecdf]
  sim_data = None
  nvars = 6 if separate_entropy else 4  # need to make this adapt
  [kvals, likelihoods] = fits.do_mcmc(mcmc_steps, nvars, data,
                                      likelihood_fn,
                                      bounds=[0, 1])
  # Skipping data simulation for now, will need to add that
  max_likelihood_index = np.argmax(likelihoods)
  return {'best_k': kvals[:, max_likelihood_index], 'sim_data' : sim_data,
          'all_ks' : kvals,
          'all_likelihood': likelihoods,
          'aicc' : eval.calculate_aicc(likelihoods[burnin:], nvars, np.sum(num_times))}

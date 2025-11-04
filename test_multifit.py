
# import common stuff
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import scipy.io as io

from matplotlib.gridspec import GridSpec

import multifit

# Read fusion times
lipid_mixing_temp = {}
fname = 'parsed/24c_parsed.mat'
mat = io.loadmat(fname)
print(mat.keys())
lipid_mixing_temp[24] = mat['fusiontimes'].flatten()
fname = 'parsed/28c_1_parsed.mat'
mat = io.loadmat(fname)
print(mat.keys())
lipid_mixing_temp[28] = mat['fusiontimes'].flatten()
fname = 'parsed/32c_parsed.mat'
mat = io.loadmat(fname)
print(mat.keys())
lipid_mixing_temp[32] = mat['fusiontimes'].flatten()
fname = 'parsed/37c_3_parsed.mat'
mat = io.loadmat(fname)
print(mat.keys())
lipid_mixing_temp[37] = mat['fusiontimes'].flatten()

#energy-space sampling
parsed_cdf = multifit.make_ecdf_inputs(lipid_mixing_temp, return_all = True, mcmc_steps=100000, burnin=10000)
plt.figure()
plt.xscale('log')
sns.kdeplot(parsed_cdf['all_Gs'][0, 1000:], color='orange')
sns.kdeplot(parsed_cdf['all_Gs'][1, 1000:], color='red')
sns.kdeplot(parsed_cdf['all_Gs'][2, 1000:], color='blue')
print(bootstrap_nmin(lipid_mixing_temp[24]))
plt.savefig('energy_space_sampling.pdf')

# rate-space sampling
burnin = 30000
#parsed_cdf_rs = multifit.make_ecdf_inputs_rs(lipid_mixing_temp, return_all = True, mcmc_steps=50000, burnin=1000)
parsed_cdf_rs = multifit.make_ecdf_inputs_rs(lipid_mixing_temp, return_all = True, mcmc_steps=200000, burnin=burnin,
                                    likelihood_fn=multifit.multi_temp_likelihood_perprot_rs)
plt.figure()
plt.xscale('log')
sns.kdeplot(parsed_cdf_rs['all_ks'][0, 10000:], color='orange')
sns.kdeplot(parsed_cdf_rs['all_ks'][1, 10000:], color='red')
sns.kdeplot(parsed_cdf_rs['all_ks'][2, 10000:], color='blue')
plt.savefig('rate_space_sampling.pdf')
print(bootstrap_nmin(lipid_mixing_temp[24]))
print(parsed_cdf_rs['best_k'])

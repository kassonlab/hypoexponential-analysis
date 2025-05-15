# Code 2024-2025 by Oyin Adenekan
# edited by Peter Kasson
""" Gillespie simulator."""

import pysb
import time
import bionetgen
import inspect
#import numpy as np
print(inspect.getfile(bionetgen))
from pysb.pathfinder import set_path

from pysb import *
from pysb.simulator.bng import BngSimulator

set_path('bng', '/usr/local/lib/python3.11/dist-packages/bionetgen/bng-linux')

def sim_gillespie_five_step(rates, num_times, upper_time_bound):

  Model()
  # set up model
  Monomer('a')
  Monomer('b')
  Monomer('c')
  Monomer('d')
  Monomer('e')
  Monomer('f')
  Parameter('a_0', 1)
  Parameter('b_0', 0)
  Parameter('c_0', 0)
  Parameter('d_0', 0)
  Parameter('e_0', 0)
  Parameter('f_0', 0)
  Initial(a(), a_0)
  Initial(b(), b_0)
  Initial(c(), c_0)
  Initial(d(), d_0)
  Initial(e(), e_0)
  Initial(f(), f_0)
  Observable('obs_a', a())
  Observable('obs_b', b())
  Observable('obs_c', c())
  Observable('obs_d', d())
  Observable('obs_e', e())
  Observable('obs_f', f())
  Parameter('k1', rates[0])
  Parameter('k2', rates[1])
  Parameter('k3', rates[2])
  Parameter('k4', rates[3])
  Parameter('k5', rates[4])
  Rule('a_to_b', a() >> b(), k1)
  Rule('b_to_c', b() >> c(), k2)
  Rule('c_to_d', c() >> d(), k3)
  Rule('d_to_e', d() >> e(), k4)
  Rule('e_to_f', e() >> f(), k5)

  start = time.time()

  dwell_times = np.zeros((num_times))
  num_failures = np.zeros((num_times))
  for idx in range(num_times):
    t = np.linspace(0, upper_time_bound, upper_time_bound)
    solver = BngSimulator(model, t)

    ran_well = False
    while not ran_well:
      sim_run = solver.run(tspan=t, n_runs=1, method='ssa')
      rxn_trace = sim_run.observables
      try:
        dwell_times[idx] = np.where(rxn_trace['obs_f'] > 0)[0][0]
        ran_well = True
      except IndexError:
        num_failures[idx] = num_failures[idx] + 1
      del sim_run

  end = time.time()
  print((end-start))

  return dwell_times

def sim_gillespie_four_step(rates, num_times, upper_time_bound):

  Model()
  # set up model
  Monomer('a')
  Monomer('b')
  Monomer('c')
  Monomer('d')
  Monomer('e')
  Parameter('a_0', 1)
  Parameter('b_0', 0)
  Parameter('c_0', 0)
  Parameter('d_0', 0)
  Parameter('e_0', 0)
  Initial(a(), a_0)
  Initial(b(), b_0)
  Initial(c(), c_0)
  Initial(d(), d_0)
  Initial(e(), e_0)
  Observable('obs_a', a())
  Observable('obs_b', b())
  Observable('obs_c', c())
  Observable('obs_d', d())
  Observable('obs_e', e())
  Parameter('k1', rates[0])
  Parameter('k2', rates[1])
  Parameter('k3', rates[2])
  Parameter('k4', rates[3])
  Rule('a_to_b', a() >> b(), k1)
  Rule('b_to_c', b() >> c(), k2)
  Rule('c_to_d', c() >> d(), k3)
  Rule('d_to_e', d() >> e(), k4)

  start = time.time()

  dwell_times = np.zeros((num_times))
  num_failures = np.zeros((num_times))
  for idx in range(num_times):
    t = np.linspace(0, upper_time_bound, upper_time_bound)
    solver = BngSimulator(model, t)

    ran_well = False
    while not ran_well:
      sim_run = solver.run(tspan=t, n_runs=1, method='ssa')
      rxn_trace = sim_run.observables
      try:
        dwell_times[idx] = np.where(rxn_trace['obs_e'] > 0)[0][0]
        ran_well = True
      except IndexError:
        num_failures[idx] = num_failures[idx] + 1
      del sim_run

  end = time.time()
  print((end-start))

  return dwell_times

# 1 step gillespie
def sim_gillespie_one_step(rates, num_times, upper_time_bound):

  Model()
  # set up model
  Monomer('a')
  Monomer('b')
  Parameter('a_0', 1)
  Parameter('b_0', 0)
  Initial(a(), a_0)
  Initial(b(), b_0)
  Observable('obs_a', a())
  Observable('obs_b', b())
  Parameter('k1', rates[0])
  Rule('a_to_b', a() >> b(), k1)

  start = time.time()

  dwell_times = np.zeros((num_times))
  for idx in range(num_times):
    t = np.linspace(0, upper_time_bound, upper_time_bound)
    solver = BngSimulator(model, t)
    sim_run = solver.run(tspan=t, n_runs=1, method='ssa')
    rxn_trace = sim_run.observables
    # dwell_times = np.array([np.where(curr_list['obs_c'] > 0)[0][0] for curr_list in rxn_traces])
    dwell_times[idx] = np.where(rxn_trace['obs_b'] > 0)[0][0]
    del sim_run
    # print('here')

  end = time.time()
  print((end-start))

  return dwell_times

# 2 step gillespie
def sim_gillespie_two_step(rates, num_times, upper_time_bound):

  Model()
  # set up model
  Monomer('a')
  Monomer('b')
  Monomer('c')
  Parameter('a_0', 1)
  Parameter('b_0', 0)
  Parameter('c_0', 0)
  Initial(a(), a_0)
  Initial(b(), b_0)
  Initial(c(), c_0)
  Observable('obs_a', a())
  Observable('obs_b', b())
  Observable('obs_c', c())
  Parameter('k1', rates[0])
  Parameter('k2', rates[1])
  Rule('a_to_b', a() >> b(), k1)
  Rule('b_to_c', b() >> c(), k2)

  start = time.time()

  dwell_times = np.zeros((num_times))
  for idx in range(num_times):
    t = np.linspace(0, upper_time_bound, upper_time_bound)
    solver = BngSimulator(model, t)
    sim_run = solver.run(tspan=t, n_runs=1, method='ssa')
    rxn_trace = sim_run.observables
    # dwell_times = np.array([np.where(curr_list['obs_c'] > 0)[0][0] for curr_list in rxn_traces])
    dwell_times[idx] = np.where(rxn_trace['obs_c'] > 0)[0][0]
    del sim_run
    # print('here')

  end = time.time()
  print((end-start))

  # parse simulations

  return dwell_times

# 3 step gillespie
def sim_gillespie_three_step(rates, num_times, upper_time_bound):

  Model()
  # set up model
  Monomer('a')
  Monomer('b')
  Monomer('c')
  Monomer('d')
  Parameter('a_0', 1)
  Parameter('b_0', 0)
  Parameter('c_0', 0)
  Parameter('d_0', 0)
  Initial(a(), a_0)
  Initial(b(), b_0)
  Initial(c(), c_0)
  Initial(d(), d_0)
  Observable('obs_a', a())
  Observable('obs_b', b())
  Observable('obs_c', c())
  Observable('obs_d', d())
  Parameter('k1', rates[0])
  Parameter('k2', rates[1])
  Parameter('k3', rates[2])
  Rule('a_to_b', a() >> b(), k1)
  Rule('b_to_c', b() >> c(), k2)
  Rule('c_to_d', c() >> d(), k3)

  start = time.time()

  dwell_times = np.zeros((num_times))
  num_failures = np.zeros((num_times))
  for idx in range(num_times):
    t = np.linspace(0, upper_time_bound, upper_time_bound)
    solver = BngSimulator(model, t)

    ran_well = False
    while not ran_well:
      sim_run = solver.run(tspan=t, n_runs=1, method='ssa')
      rxn_trace = sim_run.observables
      try:
        dwell_times[idx] = np.where(rxn_trace['obs_d'] > 0)[0][0]
        ran_well = True
      except IndexError:
        num_failures[idx] = num_failures[idx] + 1
      del sim_run

  end = time.time()
  print((end-start))

  return dwell_times

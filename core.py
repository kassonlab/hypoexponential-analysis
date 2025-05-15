# Code 2024-2025 by Oyin Adenekan
# edited by Peter Kasson
"""Core routines for hypoexponentials."""

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

# Code 2025 by Oyin Adenekan
# edited by Peter Kasson
"""Calculate overlap between samples."""

import numpy as np
from scipy.stats import gaussian_kde

def overlap_coefficient(sample_1, sample_2):

  # Estimate PDFs
  kde1 = gaussian_kde(sample_1)
  kde2 = gaussian_kde(sample_2)

  x = np.linspace(min(min(sample_1), min(sample_2)),
                  max(max(sample_1), max(sample_2)), 1000)

  # Minimum of both KDEs
  overlap = np.trapezoid(np.minimum(kde1(x), kde2(x)), x)
  return overlap


def calculate_distance(sample_1, sample_2):
  # set the overlap and distance variables
  # overlap = False
  distance = 0

  samples = np.array([sample_1, sample_2])
  samples.shape

  # determine which has higher minimum: this is the further right distribution
  which_sample_has_higher_min = np.argmax(np.min(samples, axis=1))
  # print(which_sample_has_higher_min)
  relevant_min = np.min(samples[which_sample_has_higher_min, :])
  # print(relevant_min)

  # now, i want the max of the other sample. if the max of the other sample is less than the min of the previously determined one, then there is no overlap
  max_of_other_sample = np.max(samples[1-which_sample_has_higher_min, :])
  max_of_other_sample

  if max_of_other_sample > relevant_min:
    # overlap = True
    # calculate the overlap coefficient as the distance ...
    distance = overlap_coefficient(sample_1, sample_2)
  else:
    distance = -1*np.abs(relevant_min - max_of_other_sample)
    # print(distance)

  return distance

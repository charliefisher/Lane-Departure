"""
A collection of general functions used in image processing pipelines.

Any logic that is not pipeline specific should reside in this file.
"""


import functools
from typing import Callable, Optional

import cv2
import numpy

import lane_detection.pipeline
from general import constants


def region_of_interest(pipeline, image) -> numpy.array:
  """
  Applies the region of interest mask to a given image

  :param pipeline: the pipeline the region of interest mask is applied to
  :param image: the image to apply the region of interest mask to
  :raises RuntimeError is raised if this method is called on a pipeline where the image mask is disabled
  :return: numpy.array: the image provided with the region of interest mask applied
  """

  # raise an Error if the method is called on a pipeline whose image mask is disabled
  if not pipeline.image_mask_enabled:
    raise RuntimeError('Cannot get the region of interest on a pipeline that has the image mask disabled')

  # define the region of interest as an array of arrays (since cv2.fillPoly takes an array of polygons)
  # we are essentially passing a list with a single entry where that entry is the region of interest mask
  roi = numpy.array([pipeline.region_of_interest])
  # mask is the base image to add the region of interest mask to
  mask = numpy.zeros_like(image)
  # add the region of interst mask to the base image (all black)
  n_channels = mask.shape[2] if len(mask.shape) == 3 else 1
  cv2.fillPoly(mask, roi, (255,) * n_channels)
  # mask the provided image based on the region of interest
  masked = cv2.bitwise_and(image, mask)
  # return the masked image
  return masked


class HistoricFill:
  def DEFAULT_HISTORIC_FILL_FUNC(self, x):
    lambda_ = 0.05
    k = 1

    x = x / self._fps  # convert frame number to time
    # Weibull distribution (https://en.wikipedia.org/wiki/Weibull_distribution)
    return numpy.power(lambda_, -k) * k * numpy.power(x, k-1) * numpy.exp(-numpy.power(lambda_, -k) * numpy.power(x, k))

  def __init__(self, fps: float, lane_poly_deg: int,
               store_last_n_seconds: float = 1,
               max_consecutive_autofills: int = 2,
               diff_error_autofill: Optional[numpy.array] = None,
               diff_error_incorrect: Optional[numpy.array] = None,
               historic_weighting_func: Optional[Callable[[numpy.array], numpy.array]] = None) -> None:

    # will be appended to in HistoricFill::get
    self._past_lanes = numpy.empty((0, constants.NUM_LANES_TO_DETECT, lane_poly_deg + 1), numpy.float)
    self._past_ages = numpy.empty((0,), numpy.uint16)
    self._n_consecutive_autofills = numpy.empty(constants.NUM_LANES_TO_DETECT, numpy.uint8)

    self._fps = round(fps)
    self._max_consecutive_autofills = max_consecutive_autofills
    self._max_past_lanes_size = int(round(store_last_n_seconds * self._fps))

    if historic_weighting_func is None:
      historic_weighting_func = self.DEFAULT_HISTORIC_FILL_FUNC
    self._historic_weighting_func = historic_weighting_func

    if diff_error_autofill is None:
      if lane_poly_deg == 1:
        self._diff_error_autofill = [0.075, 35]
      else:
        raise RuntimeError('No default diff_error_autofill for degree of', lane_poly_deg)

    if diff_error_incorrect is None:
      if lane_poly_deg == 1:
        self._diff_error_incorrect = [0.5, 100]
      else:
        raise RuntimeError('No default diff_error_incorrect for degree of', lane_poly_deg)


  def get(self, detected_lanes: numpy.array) -> numpy.array:
    lanes = detected_lanes

    if len(self._past_lanes) > 0:
      for lane in range(constants.NUM_LANES_TO_DETECT):
        # get the predicted future line from the past detected lines
        predicted = self._historic_average(lane)
        assert predicted is not None

        self._past_ages += 1  # increment the age of each past lane

        # if no lane was detected, use the predicted one
        if not detected_lanes[lane].any():
          lanes[lane] = predicted
        else:
          # use a weighted average of past and detected line to smooth result
          lanes[lane] = self._historic_average(lane, numpy.insert(self._past_lanes, 0, detected_lanes, axis=0),
                                               numpy.insert(self._past_ages, 0, 0))

          # calculate difference between current and most recent line
          diff = numpy.fabs(lanes[lane] - self._past_lanes[0][lane])

          # check if the difference was too large, and if so use the predicted lane
          # do not override the detected result with the predicted result if that has been done too many times
          # consecutively, except if the difference is so large, in which case the detected lane can be considered
          # incorrect
          if (self._n_consecutive_autofills[lane] < self._max_consecutive_autofills and
              numpy.any(diff > self._diff_error_autofill)) or (numpy.any(diff > self._diff_error_incorrect)):
            self._n_consecutive_autofills[lane] += 1
            lanes[lane] = predicted
          else:
            self._n_consecutive_autofills[lane] = 0

    # add current line to history
    if lanes.all():
      self._past_lanes = numpy.insert(self._past_lanes, 0, lanes, axis=0)
      self._past_ages = numpy.insert(self._past_ages, 0, 0)  # most recent lane has age of zero

    # limit size to only store last x seconds of data
    to_keep = self._past_ages < self._max_past_lanes_size
    self._past_lanes = self._past_lanes[to_keep]
    self._past_ages = self._past_ages[to_keep]
    assert len(self._past_lanes) <= self._max_past_lanes_size

    return lanes

  def _historic_average(self, lane, past_lanes: Optional[numpy.array] = None, past_ages: Optional[numpy.array] = None):
    if past_lanes is None:
      past_lanes = self._past_lanes
      past_ages = self._past_ages
    else:
      assert past_ages is not None

    assert len(past_lanes) > 0
    return numpy.average(past_lanes[:, lane], axis=0,
                         weights=self._historic_weighting_func(past_ages))



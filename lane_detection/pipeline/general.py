"""
A collection of general functions used in image processing pipelines.

Any logic that is not pipeline specific should reside in this file.
"""


from typing import Callable, Optional

import cv2
import numpy

import settings
import lane_detection.pipeline as pipeline_
from general import constants


pipeline_settings = settings.load(settings.SettingsCategories.PIPELINES, settings.PipelineSettings.GENERAL)


def region_of_interest(pipeline: 'pipeline_.Pipeline', image: numpy.array) -> numpy.array:
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


def convert_line_from_slope_intercept_to_cords(line: numpy.array) -> numpy.array:
  if not line.all():
    return None
  m, b = line
  y1 = pipeline_settings.window.height - pipeline_settings.lanes.top_offset
  y2 = pipeline_settings.lanes.bottom_offset
  x1 = int((y1 - b) / m)
  x2 = int((y2 - b) / m)
  return numpy.array([x1, y1, x2, y2], numpy.int32).reshape(4)


def convert_lines_from_slope_intercept_to_cords(lines: numpy.array) -> numpy.array:
  cord_lines = numpy.empty((0, 4), numpy.int32)
  for line in lines:
    cords = convert_line_from_slope_intercept_to_cords(line)
    cord_lines = numpy.append(cord_lines, cords, axis=0)
  return cord_lines


def display_lines(image: numpy.array, lines: numpy.array,
                  display_func: Optional[Callable[[str, numpy.array], None]] = None,
                  color: tuple[int, int, int] = (0, 255, 0), thickness: int = 10, display_overlay: bool = True,
                  overlay_name: str = 'Lines') -> numpy.array:

  if display_overlay:
    assert display_func is not None

  line_image = numpy.zeros_like(image)
  line_cords = numpy.empty((len(lines), 4), numpy.uint16)
  if lines is not None:
    for i, line in enumerate(lines):
      if line.shape[0] == 2:
        line = convert_line_from_slope_intercept_to_cords(line)
      if line is not None:
        x1, y1, x2, y2 = line.reshape(4)
        line_cords[i] = line
        # point 1, point 2, color of lines
        # line thickness in pixels
        cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)
  if display_overlay:
    display_func(overlay_name, line_image)
  return line_image, line_cords


def display_lanes(image: numpy.array, lanes: numpy.array,
                  display_func: Optional[Callable[[str, numpy.array], None]] = None,
                  color: tuple[int, int, int] = (0, 255, 0), thickness: int = 10, display_overlay: bool = True,
                  overlay_name: str = 'Lanes') -> numpy.array:

  if display_overlay:
    assert display_func is not None

  n_lanes, poly_degree = lanes.shape
  poly_degree -= 1

  if poly_degree == 1:
    line_image, line_cords = display_lines(image, lanes, display_func, display_overlay=display_overlay, overlay_name=overlay_name)
    return line_image
  else:
    print('using polylines')
    height, *_ = image.shape
    lin = numpy.linspace(pipeline_settings.lanes.top_offset,
                         height - pipeline_settings.lanes.bottom_offset,
                         num=150, dtype=numpy.uint16)
    polylines = numpy.empty((len(lanes), *lin.shape, 2), numpy.int32)
    for lane, lane_poly_coeff in enumerate(lanes):
      lane_poly = numpy.poly1d(lane_poly_coeff)
      for i, val in enumerate(lin):
        polylines[lane, i, :] = numpy.array([lane_poly(val), val], numpy.int32)

    lane_image = numpy.empty_like(image)
    for lane in range(len(lanes)):
      cv2.polylines(lane_image, numpy.int32([polylines[lane, :]]), False, color, thickness)
    if display_overlay:
     display_func(overlay_name, lane_image)
    return lane_image


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

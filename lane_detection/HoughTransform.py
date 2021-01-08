import math
import time

import cv2
import numpy
from matplotlib import pyplot as plot

import settings
from general import constants
from lane_detection.pipeline import Pipeline
from lane_detection.pipeline.general import HistoricFill, region_of_interest, display_lines, display_lanes

# import warnings
# warnings.simplefilter('ignore', numpy.RankWarning)


class HoughTransform(Pipeline):
  """
  Explain Hough Transform here
  """

  DEGREE_TO_FIT_TO_HOUGH_LINES = 1
  settings = settings.load(settings.SettingsCategories.PIPELINES, settings.PipelineSettings.HOUGH_TRANSFORM)

  def __init__(self, source: str, *,
               n_consumers: int = 0,
               should_start: bool = True,
               show_pipeline: bool = True,
               debug: bool = False):
    """
    Calls superclass __init__ (see Pipeline.__init__ for more details)

    :param source: the filename or device that the pipeline should be run on
    :param should_start: a flag indicating whether or not the pipeline should start as soon as it is instantiated
    :param show_pipeline: a flag indicating whether or not each step in the pipeline should be shown
    :param debug: a flag indicating whether or not the use is debugging the pipeline. In debug, the pipeline is
                  shown and debug statements are enabled
    """

    super().__init__(source, n_consumers=n_consumers, image_mask_enabled=True, should_start=should_start,
                     show_pipeline=show_pipeline, debug=debug)

  def _historic_lanes_weighting_function(self, x):
    k = -1.333333 - (-0.05972087 / 0.05016553) * (1 - numpy.exp(-0.05016553 * self.fps))
    b = 0.020833333 * self.fps + 1.5
    # k=-0.35
    # b=3
    return numpy.exp(k * x + b)

  def _init_pipeline(self, first_frame):
    super()._init_pipeline(first_frame)
    self._historic_fill = HistoricFill(self.fps, HoughTransform.DEGREE_TO_FIT_TO_HOUGH_LINES,
                                       store_last_n_seconds=1.5,
                                       historic_weighting_func=self._historic_lanes_weighting_function)

  def _canny(self, image):
    """
    Converts and image to grayscale, applies a gaussian blur, and then applies a canny function. The gaussian blur is
    somewhat redundant as the canny function already applies a gaussian blur (however this gives more control, allowing
    for a large gaussian kernel to be used if desired).

    :param image: the image to apply the grayscale conversion, gaussian blur and canny function on
    :return: canny: the resulting image after the operations are applied to it
    """

    # convert image to grayscale and add it to pipeline
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    self._add_knot('Grayscale', grayscale)

    # apply gaussian blur, reducing noise in grayscale image, reducing the effect of undesired lines
    # this is somehwhat redundant as canny already applies a gaussian blur but gives more controlling allowing for a
    # larger kernel to be used (if desired)
    blurred = cv2.GaussianBlur(grayscale, HoughTransform.settings.gaussian_blur.kernel_size, HoughTransform.settings.gaussian_blur.deviation)
    self._add_knot('Gaussian Blur', blurred)

    # 4 , -600
    alpha = 1
    beta = 0
    modified = cv2.convertScaleAbs(blurred, -1, alpha, beta)
    self._add_knot('Modified', modified)

    # apply canny function to image
    # canny should have ratio of 1:2 or 1:3
    # above threshold counts as leading edge
    # below threshold is rejected
    # between thresholds is accepted if it is touching a leading edge
    # canny = cv2.Canny(blurred, HoughTransform.settings.canny.lower_threshold, HoughTransform.settings.canny.upper_threshold)
    canny = cv2.Canny(modified, HoughTransform.settings.canny.lower_threshold, HoughTransform.settings.canny.upper_threshold)
    self._add_knot('Canny', canny)
    return canny

  def _filter_hough_lines_on_slope(self, dirty_lines):
    # create a matrix to hold the lines that should be kept (i.e. met the slope criterion)
    lane_lines = numpy.empty((0, HoughTransform.DEGREE_TO_FIT_TO_HOUGH_LINES + 1), numpy.float64)
    if dirty_lines is None:
      return lane_lines
    # iterate through the lanes and only keep the lines meeting the specified criteria
    for line in dirty_lines:
      # reshape the line into coordinates
      x1, y1, x2, y2 = line.reshape(4)

      # calculate slope of line (will be equal to the result of the polyfit)
      # prevents RankWarnings from using polyfit on poorly conditioned data (points that are very close - tend to be
      # horizontally detected lines)
      m_magnitude = math.fabs((y2-y1)/(x2-x1)) if x2-x1 != 0 else math.inf

      # if the slope is in the interval specified by settings.lanes.min_slope_magnitude and settings.lanes.max_slope_magnitude, add it to the
      # lanes_lines matrix
      if m_magnitude >= HoughTransform.settings.lanes.min_slope_magnitude and m_magnitude <= HoughTransform.settings.lanes.max_slope_magnitude:
        # fit a line to the coordinates and get the returned slope and intercept
        m, b, = numpy.polyfit((x1, x2), (y1, y2), HoughTransform.DEGREE_TO_FIT_TO_HOUGH_LINES)
        lane_lines = numpy.append(lane_lines, numpy.array([[m, b]]), axis=0)
    return lane_lines

  def _classify_lanes(self, lane_lines):
    num_lines, *remaining = lane_lines.shape
    labels = numpy.empty((num_lines,))

    for i in range(num_lines):
      m, b = lane_lines[i]
      labels[i] = int(m > 0)
    return labels


  def _get_closeness(self, collection):
    num_datapoints, data_dimension = collection.shape
    closeness = numpy.empty((num_datapoints, data_dimension), numpy.float64)
    for i in range(num_datapoints):
      squared_distances = numpy.empty((num_datapoints - 1, data_dimension), numpy.float64)
      other_datapoints = collection[numpy.arange(num_datapoints) != i]
      for j in range(num_datapoints-1):
        squared_distances[j] = numpy.power(numpy.fabs(collection[i] - other_datapoints[j]), 2)
      closeness[i] = numpy.sum(squared_distances, axis=0)
      closeness[i] = numpy.sqrt(closeness[i])
    return closeness


  def _apply_closeness_filter(self, collection, percent_change_threshold):
    closeness = self._get_closeness(collection)
    num_datapoints, data_dimension = collection.shape

    for i in range(data_dimension):
      indices = closeness[:, i].argsort()
      sorted_closeness_cur_column = closeness[indices][:, i]
      not_close_index = len(sorted_closeness_cur_column)
      for j in range(num_datapoints-1):
        current_closeness = sorted_closeness_cur_column[j]
        next_closeness = sorted_closeness_cur_column[j + 1]
        percent_change = (next_closeness - current_closeness) / abs(current_closeness) * 100
        # percent_change = (next_closeness - sorted_closeness_cur_column[0]) / abs(sorted_closeness_cur_column[0]) * 100
        if percent_change >= percent_change_threshold:
          not_close_index = j + 1
          break

      collection = collection[indices][0:not_close_index, :]
      closeness = closeness[indices][0:not_close_index, :]
      num_datapoints = not_close_index
    return collection

  def _run(self, frame):
    """
    Hough Transform is run on the frame to detect the lane lines

    This method handles detecting the lines, drawing them, and passing them to the lane departure algorithm

    *insert more info about Hough Transform here*

    :param frame: the current frame of the capture
    :return: void
    """

    frame = numpy.copy(frame)
    self._add_knot('Raw', frame)
    canny = self._canny(frame)

    if self._debug:
      plot.imshow(canny)
      plot.show()
      time.sleep(7.5)

    masked = region_of_interest(self, canny)
    # add the masked image to the pipeline
    self._add_knot('Region Of Interest Mask', masked)

    # row is distance accumulator in pixels
    # theta is angle accumulator in radians
    # row then theta as args
    # threshold is min num of intersections required for a line to be drawn
    # placehold array is required (empty)
    # any traced lines of length less than minLineLength are rejected
    # max line gap is the maximum distance in pixels between segmented lines which we will allow to be connected as one
    hough_lines = cv2.HoughLinesP(masked, HoughTransform.settings.hough_lines.rho, numpy.deg2rad(HoughTransform.settings.hough_lines.theta), HoughTransform.settings.hough_lines.threshold, numpy.array([]), minLineLength=HoughTransform.settings.hough_lines.min_line_length, maxLineGap=HoughTransform.settings.hough_lines.max_line_gap)

    # add the result of the hough lines to the pipeline
    hough_overlay = display_lines(frame, hough_lines, self._add_knot)
    hough_result = cv2.addWeighted(frame, 0.8, hough_overlay, 1, 0)
    self._add_knot('Hough Raw Result', hough_result)

    # filter lines based on slope and add to the pipeline
    filtered_lines = self._filter_hough_lines_on_slope(hough_lines)
    filtered_lines_overlay = display_lines(frame, filtered_lines, self._add_knot)
    filtered_lines_result = cv2.addWeighted(frame, 0.8, filtered_lines_overlay, 1, 0)
    self._add_knot('Slope Filtered Lines Result', filtered_lines_result)

    # classify each line as either right or left
    # returns a vector the same length as filtered_lines with each entry corresponding to whether the line is right or left
    line_labels = self._classify_lanes(filtered_lines)

    lanes = numpy.zeros((constants.NUM_LANES_TO_DETECT, 2))
    lines_with_closeness_filter = numpy.zeros((0, 2))

    # for both right and left lines, do the following
    for i in range(constants.NUM_LANES_TO_DETECT):
      # get the lines corresponding to correct side from filtered_lines
      lane_lines = filtered_lines[line_labels == i]
      num_lines, *remaining = lane_lines.shape

      # add the classified lines to the pipeline
      classified_lines_overlay = display_lanes(frame, lane_lines, self._add_knot, display_overlay=False)
      classified_lines_result = cv2.addWeighted(frame, 0.8, classified_lines_overlay, 1, 0)
      self._add_knot('{side} Lane Lines'.format(side='Left' if i == 0 else 'Right'), classified_lines_result)

      # for more than 2 lines, apply the closeness filter - less than two does not work
      if num_lines > 2:
        lane_lines = self._apply_closeness_filter(lane_lines, 15)
        lines_with_closeness_filter = numpy.append(lines_with_closeness_filter, lane_lines, axis=0)
      # if any lanes were detected, average the result
      if num_lines != 0:
        # averaged_lane = numpy.average(lane_lines[labels == i], axis=0) # old line when closeness filter did not exist
        averaged_lane = numpy.average(lane_lines, axis=0).reshape(1, 2)
        # add the cluster's average lane to the matrix of lane lines
        lanes[i] = averaged_lane

    closeness_filter_applied_overlay = display_lanes(frame, lines_with_closeness_filter, self._add_knot)
    closeness_filter_applied_result = cv2.addWeighted(frame, 0.8, closeness_filter_applied_overlay, 1, 0)
    self._add_knot('Closeness Filter Result', closeness_filter_applied_result)

    detected_lanes_overlay = display_lanes(frame, lanes, self._add_knot)
    detected_lanes_result = cv2.addWeighted(frame, 0.8, detected_lanes_overlay, 1, 0)
    self._add_knot('Detected Lanes Result', detected_lanes_result)

    lanes = self._historic_fill.get(numpy.array(lanes))
    lane_image = display_lanes(frame, lanes, self._add_knot, overlay_name='Historic Filtered')
    detected_lanes_result = cv2.addWeighted(frame, 0.75, lane_image, 1, 0)
    self._add_knot('Historic Filtered Result', detected_lanes_result)

    self._add_lanes(lanes)

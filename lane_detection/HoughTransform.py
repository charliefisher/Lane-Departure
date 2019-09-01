from configparser import ConfigParser
from os import path
import cv2
import numpy
import math
import time
from matplotlib import pyplot as plot

from lane_detection.Pipeline import Pipeline


class HoughTransform(Pipeline):
  """
  Explain Hough Transform here
  """

  # set up config file reader
  __config = ConfigParser(allow_no_value=True)
  __config.read(path.join(path.dirname(__file__), r'./HoughTransform.config'))
  # set up static variables from config file
  MIN_SLOPE_MAGNITUDE = float(__config['lines']['min_slope_magnitude'])
  NUM_LANES_TO_DETECT = int(__config['lines']['num_lanes_to_detect'])
  K_MEANS_NUM_ATTEMPTS = 10

  def __init__(self, source, should_start, show_pipeline, debug):
    """
    Calls superclass __init__ (see Pipeline.__init__ for more details)

    :param source: the filename or device that the pipeline should be run on
    :param should_start: a flag indicating whether or not the pipeline should start as soon as it is instantiated
    :param show_pipeline: a flag indicating whether or not each step in the pipeline should be shown
    :param debug: a flag indicating whether or not the use is debugging the pipeline. In debug, the pipeline is
                  shown and debug statements are enabled
    """

    super().__init__(source, should_start, show_pipeline, debug)

  def _canny(self, image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    self._add_knot('Grayscale', grayscale)
    # 5x5 kernel
    # deviation of 0
    # reduces noise in grayscale image
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    self._add_knot('Gaussian Blur', blurred)
    # canny auto applies guassian blur anyhow
    # canny should have ratio of 1:2 or 1:3 above counts as leading edge, below is rejected between is accepted if it is touching a leading edge
    canny = cv2.Canny(blurred, 50, 150)
    self._add_knot('Canny', canny)
    return canny

  def _region_of_interest(self, image):
    # height = image.shape[0]
    # width = image.shape[1]
    roi = numpy.array([
      [(100, 500), (1250, 500), (800, 200), (500, 200)]
    ])
    mask = numpy.zeros_like(image)
    # 255 indicates roi is all white
    cv2.fillPoly(mask, roi, 255)
    # mask the provided image based on the roi
    masked = cv2.bitwise_and(image, mask)
    self._add_knot('Region Of Interest Mask', masked)
    return masked

  def _display_lines(self, image, lines):
    line_image = numpy.zeros_like(image)
    if lines is not None:
      # print('lines', lines)
      for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        # point 1, point 2, color of lines
        # line thickness in pixels
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    self._add_knot('Lines', line_image)
    return line_image

  def _lines_as_slope_intercept_to_cords(self, lines):
    cord_lines = numpy.empty((0, 4), numpy.int32)
    for line in lines:
      m, b = line
      y1 = 500
      y2 = 200
      x1 = int((y1 - b) / m)
      x2 = int((y2 - b) / m)
      cord_lines = numpy.append(cord_lines, numpy.array([x1, y1, x2, y2]).reshape((1, 4)), axis=0)
    return cord_lines


  def _clean_lines(self, dirty_lines):
    DEGREE_TO_FIT_TO_HOUGH_LINES = 1
    # create a matrix to hold the lines that should be kept (i.e. met the MIN_SLOPE_MAGNITUDE criteria)
    lane_lines = numpy.empty((0, DEGREE_TO_FIT_TO_HOUGH_LINES + 1), numpy.float64)
    # catch case where no lines were detected
    if dirty_lines is None:
      return lane_lines
    for line in dirty_lines:
      # reshape the line into coordinates
      x1, y1, x2, y2 = line.reshape(4)
      # fit a line to the coordinates and get the returned slope and intercept
      m, b = numpy.polyfit((x1, x2), (y1, y2), DEGREE_TO_FIT_TO_HOUGH_LINES)
      # if the slope is greater than the MIN_SLOPE_MAGNITUDE, add it to the lanes_lines matrix
      if math.fabs(m) >= HoughTransform.MIN_SLOPE_MAGNITUDE:
        lane_lines = numpy.append(lane_lines, numpy.array([[m, b]]), axis=0)

    # create a matrix of lanes that each cluster's average will be added to
    lanes = numpy.empty((0, DEGREE_TO_FIT_TO_HOUGH_LINES + 1), numpy.float64)
    # catch case where number of lines detected is less than the number of lines to detect
    if len(lane_lines) < HoughTransform.NUM_LANES_TO_DETECT:
      return self._lines_as_slope_intercept_to_cords(lane_lines)
    # convert lane_lines to numpy.float32
    lane_lines = numpy.float32(lane_lines)

    # define criteria for k means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 500, 1.0)
    # criteria = (cv2.TERM_CRITERIA_EPS, 0, 0.5)
    # run k means
    compactness, labels, centers = cv2.kmeans(lane_lines, HoughTransform.NUM_LANES_TO_DETECT, None, criteria, HoughTransform.K_MEANS_NUM_ATTEMPTS, cv2.KMEANS_RANDOM_CENTERS)
    # flatten labels so they can be used for boolean indexing
    labels = labels.flatten()
    for i in range(HoughTransform.NUM_LANES_TO_DETECT):
      # get the average of each cluster
      averaged_lane = numpy.average(lane_lines[labels == i], axis=0)
      # reshape it to a 1x2 matrix
      averaged_lane = averaged_lane.reshape(1, 2)
      # add the cluster's average lane to the matrix of lane lines
      lanes = numpy.append(lanes, averaged_lane, axis=0)
    return self._lines_as_slope_intercept_to_cords(lanes)


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
    masked = self._region_of_interest(canny)
    # row is distance accumulator in pixels
    # theta is angle accumulator in radians
    # row then theta as args
    # threshold is min num of intersections required for a line to be drawn
    # placehold array is required (empty)
    # any traced lines of length less than minLineLength are rejected
    # max line gap is the maximum distance in pixels between segmented lines which we will allow to be connected as one
    hough_lines = cv2.HoughLinesP(masked, 2, numpy.pi/180, 100, numpy.array([]), minLineLength=40, maxLineGap=5)

    hough_overlay = self._display_lines(frame, hough_lines)
    hough_result_raw = cv2.addWeighted(frame, 0.8, hough_overlay, 1, 0)
    self._add_knot('Hough Raw Result', hough_result_raw)

    cleaned_lines = self._clean_lines(hough_lines)
    # print('cleaned_lines', cleaned_lines)
    lines_overlay = self._display_lines(frame, cleaned_lines)
    # number is weight of first array then second
    # last value is gamma value that is added to sum
    result = cv2.addWeighted(frame, 0.8, lines_overlay, 1, 0)
    self._add_knot('Final Result', result)

    # plot.imshow(canny)
    # plot.show()
    # time.sleep(10)

    if self._show_pipeline:
      self._display_pipeline()

from configparser import ConfigParser
from os import path
import re
import cv2
import numpy
import math
import time
from matplotlib import pyplot as plot

from lane_detection.Pipeline.Pipeline import Pipeline
from lane_detection.Pipeline.general import region_of_interest

# import warnings
# warnings.simplefilter('ignore', numpy.RankWarning)


class HoughTransform(Pipeline):
  """
  Explain Hough Transform here
  """

  DEGREE_TO_FIT_TO_HOUGH_LINES = 1

  # set up config file reader
  __config = ConfigParser(allow_no_value=True)
  __config.read(path.join(path.dirname(__file__), r'./HoughTransform.config'))
  # set up static variables from config file
  MIN_SLOPE_MAGNITUDE = float(__config['lanes']['min_slope_magnitude'])
  MAX_SLOPE_MAGNITUDE = float(__config['lanes']['max_slope_magnitude'])
  NUM_LANES_TO_DETECT = int(__config['lanes']['num_to_detect'])

  GAUSSIAN_BLUR_KERNEL_SIZE = tuple(map(int, re.sub('\(|\)| ', '', __config['gaussian blur']['kernel_size']).split(',')))
  GAUSSIAN_BLUR_DEVIATION = float(__config['gaussian blur']['deviation'])

  CANNY_LOWER_THRESHOLD = int(__config['canny']['lower_threshold'])
  CANNY_UPPER_THRESHOLD = int(__config['canny']['upper_threshold'])

  HOUGH_LINES_RHO_PIXELS = int(__config['hough lines']['rho'])
  HOUGH_LINES_THETA_DEGREES = float(__config['hough lines']['theta'])
  HOUGH_LINES_THRESHOLD = int(__config['hough lines']['threshold'])
  HOUGH_LINES_MIN_LINE_LENGTH = int(__config['hough lines']['min_line_length'])
  HOUGH_LINES_MAX_LINE_GAP = int(__config['hough lines']['max_line_gap'])

  K_MEANS_MAX_ITER = int(__config['k means']['max_iter'])
  K_MEANS_EPSILON = float(__config['k means']['epsilon'])
  K_MEANS_NUM_ATTEMPTS = int(__config['k means']['num_attempts'])

  def __init__(self, source, should_start, show_pipeline, debug):
    """
    Calls superclass __init__ (see Pipeline.__init__ for more details)

    :param source: the filename or device that the pipeline should be run on
    :param should_start: a flag indicating whether or not the pipeline should start as soon as it is instantiated
    :param show_pipeline: a flag indicating whether or not each step in the pipeline should be shown
    :param debug: a flag indicating whether or not the use is debugging the pipeline. In debug, the pipeline is
                  shown and debug statements are enabled
    """

    self.__past_detected = numpy.empty((0, HoughTransform.NUM_LANES_TO_DETECT, 2))
    self.__consecutive_overrides_of_detected_line = numpy.zeros((1, HoughTransform.NUM_LANES_TO_DETECT))
    super().__init__(source, should_start, show_pipeline, debug, True)

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
    blurred = cv2.GaussianBlur(grayscale, HoughTransform.GAUSSIAN_BLUR_KERNEL_SIZE, HoughTransform.GAUSSIAN_BLUR_DEVIATION)
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
    # canny = cv2.Canny(blurred, HoughTransform.CANNY_LOWER_THRESHOLD, HoughTransform.CANNY_UPPER_THRESHOLD)
    canny = cv2.Canny(modified, HoughTransform.CANNY_LOWER_THRESHOLD, HoughTransform.CANNY_UPPER_THRESHOLD)
    self._add_knot('Canny', canny)
    return canny

  def _convert_line_from_slope_intercept_to_cords(self, line):
    if not line.all():
      return None
    m, b = line
    y1 = 500
    y2 = 200
    x1 = int((y1 - b) / m)
    x2 = int((y2 - b) / m)
    return numpy.array([x1, y1, x2, y2], numpy.int32).reshape(4)

  def _convert_lines_from_slope_intercept_to_cords(self, lines):
    cord_lines = numpy.empty((0, 4), numpy.int32)
    for line in lines:
      cords = self._convert_line_from_slope_intercept_to_cords(line)
      cord_lines = numpy.append(cord_lines, cords, axis=0)
    return cord_lines

  def _display_lines(self, image, lines, display_overlay=True, overlay_name='Lines'):
    line_image = numpy.zeros_like(image)
    if lines is not None:
      for line in lines:
        if line.shape[0] == 2:
          line = self._convert_line_from_slope_intercept_to_cords(line)
        if line is not None:
          x1, y1, x2, y2 = line.reshape(4)
          # point 1, point 2, color of lines
          # line thickness in pixels
          cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    if display_overlay:
      self._add_knot(overlay_name, line_image)
    return line_image


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

      # if the slope is in the interval specified by MIN_SLOPE_MAGNITUDE and MAX_SLOPE_MAGNITUDE, add it to the
      # lanes_lines matrix
      if m_magnitude >= HoughTransform.MIN_SLOPE_MAGNITUDE and m_magnitude <= HoughTransform.MAX_SLOPE_MAGNITUDE:
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

  def _get_weighted_historic_average(self, historic_lines, column):
    def weighting_function(fps, x):
      k = -1.333333 - (-0.05972087 / 0.05016553) * (1 - math.pow(math.e, (-0.05016553 * fps)))
      b = 0.020833333*fps+1.5
      # k=-0.35
      # b=3
      return numpy.exp(k*x+b)


    num_past_stored, *remaining = historic_lines.shape
    if num_past_stored == 0:
      return None

    return numpy.average(historic_lines[:, column, :], axis=0, weights=weighting_function(self._fps, numpy.arange(historic_lines.shape[0])))

  def _historic_fill(self, column):
    # TODO: do historic fill using derivatives so it guesses the next line then takes the weighted historic average
    return self._get_weighted_historic_average(self.__past_detected, column)





    # # convert lane_lines to numpy.float32
    # lane_lines = numpy.float32(lane_lines)
    #
    # type = 0
    # if HoughTransform.K_MEANS_EPSILON > 0:
    #   type += cv2.TERM_CRITERIA_EPS
    # if HoughTransform.K_MEANS_MAX_ITER > 0:
    #   type += cv2.TERM_CRITERIA_MAX_ITER
    # # define criteria for k means
    # criteria = (type, HoughTransform.K_MEANS_MAX_ITER, HoughTransform.K_MEANS_EPSILON)
    # # criteria = (cv2.TERM_CRITERIA_EPS, 0, 0.5)
    # # run k means
    # compactness, labels, centers = cv2.kmeans(lane_lines, HoughTransform.NUM_LANES_TO_DETECT, None, criteria, HoughTransform.K_MEANS_NUM_ATTEMPTS, cv2.KMEANS_RANDOM_CENTERS)
    # # flatten labels so they can be used for boolean indexing
    # labels = labels.flatten()

    # for i in range(HoughTransform.NUM_LANES_TO_DETECT):
    #   lines_in_cluster = lane_lines[labels == i]
      # print('labeled', lines_in_cluseter)
      # mu, sigma = numpy.mean(lines_in_cluseter, axis=0), numpy.std(lines_in_cluseter, axis=0, ddof=1)
      # print('mu, sigma', mu, sigma)
      # print(sigma[~numpy.isnan(sigma)])
      # print('simga', numpy.isfinite(sigma[0]) and numpy.isfinite(sigma[1]))
      # if numpy.isfinite(sigma[0]) and numpy.isfinite(sigma[1]):
      #   print('filter')
      #   lines_in_cluseter = lines_in_cluseter[numpy.all(numpy.abs(lines_in_cluseter - mu) < 1.0*sigma, axis=1)]
      #   compactness, labels, centers = cv2.kmeans(lines_in_cluseter, 1, None, criteria, HoughTransform.K_MEANS_NUM_ATTEMPTS, cv2.KMEANS_RANDOM_CENTERS)
      #   print('lables', labels)
      #   print('lines_in_cluster', lines_in_cluseter)
      #   print(len(labels) == len(lines_in_cluseter))
      #   lines_in_cluseter = lines_in_cluseter[labels == i]


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
    hough_lines = cv2.HoughLinesP(masked, HoughTransform.HOUGH_LINES_RHO_PIXELS, numpy.deg2rad(HoughTransform.HOUGH_LINES_THETA_DEGREES), HoughTransform.HOUGH_LINES_THRESHOLD, numpy.array([]), minLineLength=HoughTransform.HOUGH_LINES_MIN_LINE_LENGTH, maxLineGap=HoughTransform.HOUGH_LINES_MAX_LINE_GAP)

    # add the result of the hough lines to the pipeline
    hough_overlay = self._display_lines(frame, hough_lines)
    hough_result = cv2.addWeighted(frame, 0.8, hough_overlay, 1, 0)
    self._add_knot('Hough Raw Result', hough_result)

    # filter lines based on slope and add to the pipeline
    filtered_lines = self._filter_hough_lines_on_slope(hough_lines)
    filtered_lines_overlay = self._display_lines(frame, filtered_lines)
    filtered_lines_result = cv2.addWeighted(frame, 0.8, filtered_lines_overlay, 1, 0)
    self._add_knot('Slope Filtered Lines Result', filtered_lines_result)

    # classify each line as either right or left
    # returns a vector the same length as filtered_lines with each entry corresponding to whether the line is right or left
    line_labels = self._classify_lanes(filtered_lines)

    lanes = numpy.zeros((HoughTransform.NUM_LANES_TO_DETECT, 2))
    lines_with_closeness_filter = numpy.zeros((0, 2))

    # for both right and left lines, do the following
    for i in range(HoughTransform.NUM_LANES_TO_DETECT):
      # get the lines corresponding to correct side from filtered_lines
      lane_lines = filtered_lines[line_labels == i]
      num_lines, *remaining = lane_lines.shape

      # add the classified lines to the pipeline
      classified_lines_overlay = self._display_lines(frame, lane_lines, display_overlay=False)
      classified_lines_result = cv2.addWeighted(frame, 0.8, classified_lines_overlay, 1, 0)
      self._add_knot('{side} Lane Lines'.format(side='Left' if i == 0 else 'Right'), classified_lines_result)

      # for more than 2 lines, apply the closeness filter - less than two does not work
      if num_lines > 2:
        lane_lines = self._apply_closeness_filter(lane_lines, 15)
        lines_with_closeness_filter = numpy.append(lines_with_closeness_filter, lane_lines, axis=0)
      # if any lanes were detected, average the result
      if num_lines != 0:
        # averaged_lane = numpy.average(lane_lines[labels == i], axis=0) #old line when closeness filter did not exist
        averaged_lane = numpy.average(lane_lines, axis=0).reshape(1, 2)
        # add the cluster's average lane to the matrix of lane lines
        lanes[i] = averaged_lane

    closeness_filter_applied_overlay = self._display_lines(frame, lines_with_closeness_filter)
    closeness_filter_applied_result = cv2.addWeighted(frame, 0.8, closeness_filter_applied_overlay, 1, 0)
    self._add_knot('Closeness Filter Result', closeness_filter_applied_result)

    detected_lanes_overlay = self._display_lines(frame, lanes)
    detected_lanes_result = cv2.addWeighted(frame, 0.8, detected_lanes_overlay, 1, 0)
    self._add_knot('Detected Lanes Result', detected_lanes_result)

    past_lines = numpy.empty((1, HoughTransform.NUM_LANES_TO_DETECT, 2))
    for i in range(HoughTransform.NUM_LANES_TO_DETECT):
      # get the predicted future line from the past detected lines
      historic_fill = self._historic_fill(column=i)

      # if no lane was detected, use the predicted one
      if not lanes[i].any():
        lanes[i] = historic_fill
      else:
        # use a weighted average of past and detected line to smooth result
        lanes[i] = self._get_weighted_historic_average(numpy.insert(self.__past_detected, 0, numpy.array([lanes]), axis=0), column=i)
        # compare the detected lane with what we expect to find
        # do not override the detected result with the predicted result if that has been done too many times consecutively
        if self.__consecutive_overrides_of_detected_line[0][i] < 2:
          if historic_fill is not None:
            # calculate difference between predicted and detected lines
            last_and_cur_diff = numpy.fabs(lanes[i] - self.__past_detected[0][i])
            # check if the difference was too large, and if so use the predicted line
            if last_and_cur_diff[0] > 0.075 or last_and_cur_diff[1] > 35:
              self.__consecutive_overrides_of_detected_line[0][i] += 1
              lanes[i] = historic_fill
        else:
          self.__consecutive_overrides_of_detected_line[0][i] = 0

      past_lines[0][i] = lanes[i]

    if past_lines[0].all():
      self.__past_detected = numpy.insert(self.__past_detected, 0, past_lines, axis=0)

    if self.__past_detected.shape[0] > int(round(1.5*self._fps)):
      self.__past_detected = numpy.delete(self.__past_detected, self.__past_detected.shape[0]-1, axis=0)

    fill_and_average_overlay = self._display_lines(frame, lanes)
    fill_and_average_result = cv2.addWeighted(frame, 0.8, fill_and_average_overlay, 1, 0)
    self._add_knot('Fill And Average Lanes Result', fill_and_average_result)

    if self._show_pipeline:
      self._display_pipeline()

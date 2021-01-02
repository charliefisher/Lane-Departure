import time
from typing import Union

import cv2
import numpy
from matplotlib import pyplot as plot

import settings
from lane_detection.pipeline import Pipeline
from lane_detection.pipeline.general import region_of_interest


class HistogramPeakDetection(Pipeline):
  """
   Explain Histogram Peak Detection here
   """

  DEGREE_TO_FIT_TO_LINES = 5
  settings = settings.load(settings.SettingsCategories.PIPELINES, settings.PipelineSettings.HISTOGRAM_PEAK_DETECTION)

  def __init__(self, source: str, *,
               should_start: bool = True,
               show_pipeline: bool = True,
               debug: bool = False):
    """
    Calls superclass __init__ (see Pipeline.__init__ for more details)

    :param source: the filename or device that the pipeline should be run on
    :param should_start: a flag indicating whether or not the pipeline should start as soon as it is instantiated
    :param debug: a flag indicating whether or not the use is debugging the pipeline. In debug, the pipeline is
                  shown and debug statements are enabled
    """

    super().__init__(source, image_mask_enabled=True, should_start=should_start,
                     show_pipeline=show_pipeline, debug=debug)

  def _add_knot(self, name: str, image: numpy.array, hls: bool = True):
    if hls:
      image = cv2.cvtColor(image, cv2.COLOR_HLS2RGB)
    return super()._add_knot(name, image)

  def filter_thresholds(self, image: numpy.array, thresholds: Union[tuple[int, int], list[tuple[int, int]]])\
      -> numpy.array:
    if isinstance(thresholds, tuple):
      thresholds = [thresholds]

    filter_mask = numpy.zeros_like(image)
    for threshold in thresholds:
      lower_threshold, upper_threshold = threshold
      filter_mask[(lower_threshold <= image) & (image <= upper_threshold)] = 255
    return filter_mask

  def fill_filter_close_regions(self, filter_image: numpy.array, max_fill_distance: int = 20) -> numpy.array:
    filter = numpy.copy(filter_image)
    for row_index, row in enumerate(filter):
      last_filled_index = None
      for pixel_index, pixel in enumerate(row):
        if pixel == 255:
          if last_filled_index is not None and pixel_index - last_filled_index <= max_fill_distance:
            filter[row_index, last_filled_index + 1:pixel_index - 1] = 255
          last_filled_index = pixel_index
    return filter

  def get_histogram_peak_points(self, detected_image: numpy.array, window_height: int = 10)\
      -> tuple[numpy.array, numpy.array]:

    height, width, *_ = detected_image.shape
    height -= HistogramPeakDetection.settings.lanes.bottom_offset + HistogramPeakDetection.settings.lanes.top_offset
    if height % window_height != 0:
      raise FloatingPointError("window_height must evenly divide the detected_image height")
    num_vertical_windows = height // window_height

    # 3rd dimension: 0th entry is left side, 1st entry is right side
    vertical_averages = numpy.empty((2, num_vertical_windows, width // 2), numpy.float)
    for i in range(num_vertical_windows):
      window = detected_image[HistogramPeakDetection.settings.lanes.top_offset + i * window_height:
                              HistogramPeakDetection.settings.lanes.top_offset + (i+1) * window_height + 1, :]
      vertical_window_average = numpy.average(window, axis=0)
      vertical_averages[0, i, :] = vertical_window_average[0 : width // 2]
      vertical_averages[1, i, :] = vertical_window_average[width // 2 : ]

    window_maxes = numpy.iinfo(numpy.uint16).max * numpy.ones((2, num_vertical_windows), numpy.uint16)
    for i in range(num_vertical_windows):
      for lane in range(2):
        if numpy.max(vertical_averages[lane, i, :]) >= 150:
          window_maxes[lane, i] = numpy.argmax(vertical_averages[lane, i, :])

    if self.debug:
      plot.bar(numpy.arange(width), numpy.average(detected_image, axis=0))
      plot.show()
      time.sleep(7.5)

    left_points: list[tuple[int, int]] = []
    right_points: list[tuple[int, int]] = []
    for i in range(num_vertical_windows):
      y_cord = HistogramPeakDetection.settings.lanes.top_offset + i * window_height + window_height // 2
      if window_maxes[0, i] != numpy.iinfo(numpy.uint16).max:
        left_points.append((window_maxes[0, i], y_cord))
      if window_maxes[1, i] != numpy.iinfo(numpy.uint16).max:
        right_points.append((window_maxes[1, i] + width // 2, y_cord))

    return numpy.array(left_points), numpy.array(right_points)

  def classify_points(self, points, labels = None):
    points = numpy.float32(points)
    # # convert lane_lines to numpy.float32
    # lane_lines = numpy.float32(lane_lines)

    type = 0
    if HistogramPeakDetection.settings.k_means.epsilon > 0:
      type += cv2.TERM_CRITERIA_EPS
    if HistogramPeakDetection.settings.k_means.max_iter > 0:
      type += cv2.TERM_CRITERIA_MAX_ITER
    # define criteria for k means
    criteria = (type, HistogramPeakDetection.settings.k_means.max_iter, HistogramPeakDetection.settings.k_means.epsilon)
    # run k means
    compactness, labels, centers = cv2.kmeans(points, HistogramPeakDetection.settings.lanes.num_to_detect, labels,
                                              criteria, HistogramPeakDetection.settings.k_means.num_attempts,
                                              # cv2.KMEANS_RANDOM_CENTERS)
                                              cv2.KMEANS_USE_INITIAL_LABELS)
    # flatten labels so they can be used for boolean indexing
    labels = labels.flatten()

    classified_points = HistogramPeakDetection.settings.lanes.num_to_detect * [None]
    for i in range(HistogramPeakDetection.settings.lanes.num_to_detect):
      points_in_cluster = points[labels == i]
      mu_x, mu_y = numpy.mean(points_in_cluster, axis=0)
      # sigma = numpy.std(lines_in_cluster, axis=0, ddof=1)
      classified_points[int(not mu_x < 600)] = points_in_cluster.astype(numpy.int32)

    return tuple(classified_points)


  def fit_lane_to_points(self, points, polynomial_degree: int = 4)\
      -> tuple[numpy.array, numpy.array]:

    left_points, right_points = points
    # fit a polynomial for each lane and return the result
    left_lane = numpy.polyfit(left_points[:, 0], left_points[:, 1], deg=polynomial_degree)
    right_lane = numpy.polyfit(right_points[:, 0], right_points[:, 1], deg=polynomial_degree)
    return left_lane, right_lane

  def _display_points(self, image, points, radius: int = 5, color: tuple[int, int, int] = (255, 0, 255),
                      display_overlay: bool = True, overlay_name: str = 'Points') -> numpy.array:
    points_image = numpy.zeros_like(image)
    for point in points:
      cv2.circle(points_image, tuple(point), radius, color, -1)
    if display_overlay:
      self._add_knot(overlay_name, points_image, hls=False)
    return points_image

  def _display_lanes(self, image, lanes, color: tuple[int, int, int] = (0, 255, 0), thickness: int = 10,
                     display_overlay: bool = True, overlay_name: str = 'Lanes') -> numpy.array:

    height, *_ = image.shape
    lin = numpy.linspace(HistogramPeakDetection.settings.lanes.top_offset,
                         height - HistogramPeakDetection.settings.lanes.bottom_offset,
                         num=150, dtype=numpy.uint16)
    polylines = numpy.empty((2, *lin.shape, 2), numpy.int32)
    for lane, lane_poly_coeff in enumerate(lanes):
      lane_poly = numpy.poly1d(lane_poly_coeff)
      for i, val in enumerate(lin):
        polylines[lane, i, :] = numpy.array([lane_poly(val), val], numpy.int32)

    lane_image = numpy.empty_like(image)
    for i in range(2):
      cv2.polylines(lane_image, numpy.int32([polylines[i, :]]), False, color, thickness)
    if display_overlay:
      self._add_knot(overlay_name, lane_image, hls=False)
    return lane_image


  def _run(self, frame):
    """
    Histogram Peak Detection is run on the frame to detect the lane lines

    This method handles detecting the lines, drawing them, and passing them to the lane departure algorithm

    *insert more info about Histogram Peak Detection here*

    :param frame: the current frame of the capture
    :return: void
    """

    frame = numpy.copy(frame)
    self._add_knot('Raw', frame, hls=False)

    hls = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS)

    # apply gaussian blur, reducing noise in grayscale image, reducing the effect of undesired lines
    # this is somehwhat redundant as canny already applies a gaussian blur but gives more controlling allowing for a
    # larger kernel to be used (if desired)
    blurred = cv2.GaussianBlur(hls, HistogramPeakDetection.settings.gaussian_blur.kernel_size,
                               HistogramPeakDetection.settings.gaussian_blur.deviation)
    self._add_knot('Gaussian Blur', blurred)

    hls_64f = blurred.astype(numpy.float)
    h_channel = hls_64f[:, :, 0]
    l_channel = hls_64f[:, :, 1]
    s_channel = hls_64f[:, :, 2]

    # Negative slopes will be ignored if using uint8, thus, we run the sobel filter over a float image and convert later
    sobelx_64f = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=5)
    abs_sobelx_64f = numpy.absolute(sobelx_64f)
    sobelx = numpy.uint8(abs_sobelx_64f)
    sobelx_scaled = numpy.uint8(255 * (sobelx / numpy.max(sobelx)))
    self._add_knot('Sobel X', sobelx_scaled, hls=False)

    sobel_filter_mask = self.filter_thresholds(sobelx_scaled, (125, 255))
    self._add_knot('Sobel  Filter Mask', sobel_filter_mask, hls=False)
    sobel_filtered = cv2.bitwise_and(sobelx_scaled, sobel_filter_mask)
    self._add_knot('Sobel  Filtered', sobel_filtered, hls=False)

    s_channel = s_channel.astype(numpy.uint8)
    self._add_knot('Saturation  Channel', s_channel, hls=False)
    saturation_filter_mask = self.filter_thresholds(s_channel, (10, 50))
    self._add_knot('Saturation  Filter Mask', saturation_filter_mask, hls=False)
    saturation_filtered = cv2.bitwise_and(s_channel, saturation_filter_mask)
    self._add_knot('Saturation  Filtered', saturation_filtered, hls=False)

    combined_filter = sobel_filter_mask & saturation_filter_mask
    self._add_knot('Combined Filters', combined_filter, hls=False)

    contours = cv2.findContours(combined_filter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    for contour in contours:
      cv2.drawContours(combined_filter, [contour], -1, (255, 255, 255), -1)

    self._add_knot('Combined Filters FILLED CONTOURS', combined_filter, hls=False)

    masked = region_of_interest(self, combined_filter)
    self._add_knot('Region Of Interest Mask', masked, hls=False)

    points = self.get_histogram_peak_points(masked)

    # labels = numpy.append(numpy.zeros((len(points[0]),), numpy.int32), numpy.ones((len(points[1]),), numpy.int32),
    #                       axis=0)
    # points = self.classify_points([point for side in points for point in side], labels)

    # display the detected points on the imagew
    masked_image_rgb = cv2.cvtColor(masked, cv2.COLOR_GRAY2RGB)
    left_points, right_points = points

    points_image = self._display_points(masked_image_rgb, left_points)
    detected_points_result = cv2.addWeighted(masked_image_rgb, 0.5, points_image, 1, 0)
    self._add_knot('Detected Left Points Result', detected_points_result, hls=False)

    points_image = self._display_points(masked_image_rgb, right_points)
    detected_points_result = cv2.addWeighted(masked_image_rgb, 0.5, points_image, 1, 0)
    self._add_knot('Detected Right Points Result', detected_points_result, hls=False)


    points_image = self._display_points(masked_image_rgb, [point for side in points for point in side])
    detected_points_result = cv2.addWeighted(masked_image_rgb, 0.5, points_image, 1, 0)
    self._add_knot('Detected Points Result', detected_points_result, hls=False)


    lanes = self.fit_lane_to_points(points)

    lane_image = self._display_lanes(frame, lanes)
    detected_lanes_result = cv2.addWeighted(frame, 0.5, lane_image, 1, 0)
    self._add_knot('Detected Lanes Result', detected_lanes_result, hls=False)



    # filled_combined_filter = self.fill_filter_close_regions(masked)
    # # filled_combined_filter = masked
    # self._add_knot('Combined Filters Custom Fill', filled_combined_filter, hls=False)

    # x threshold
    # saturation threshold

    # combine thresholds into binary



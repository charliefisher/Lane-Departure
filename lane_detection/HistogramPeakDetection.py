import time
from typing import Union

import cv2
import numpy
from matplotlib import pyplot as plot

import settings
from lane_detection.pipeline import Pipeline
from lane_detection.pipeline.general import HistoricFill, region_of_interest
from general.config_dict import config_dict


class HistogramPeakDetection(Pipeline):
  """
   Explain Histogram Peak Detection here
   """

  DEGREE_TO_FIT_TO_LINES: int = 1
  settings: config_dict = settings.load(settings.SettingsCategories.PIPELINES,
                                        settings.PipelineSettings.HISTOGRAM_PEAK_DETECTION)

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

  def _init_pipeline(self, first_frame):
    super()._init_pipeline(first_frame)
    self._historic_fill = HistoricFill(self.fps, HistogramPeakDetection.DEGREE_TO_FIT_TO_LINES,
                                       max_consecutive_autofills=5)

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

  def get_histogram_peak_points(self, detected_image: numpy.array,
                                window_height: int = settings.sliding_window.height,
                                n_max_per_window: int = settings.sliding_window.n_max_per_window)\
      -> tuple[numpy.array, numpy.array]:

    height, width, *_ = detected_image.shape
    height -= HistogramPeakDetection.settings.lanes.bottom_offset + HistogramPeakDetection.settings.lanes.top_offset
    if height % window_height != 0:
      raise FloatingPointError("window_height must evenly divide the detected_image height")
    num_vertical_windows = height // window_height

    # 3rd dimension: 0th entry is left side, 1st entry is right side
    vertical_averages = numpy.empty((HistogramPeakDetection.settings.lanes.num_to_detect, num_vertical_windows,
                                     width // HistogramPeakDetection.settings.lanes.num_to_detect),
                                    numpy.float)
    for i in range(num_vertical_windows):
      window = detected_image[HistogramPeakDetection.settings.lanes.top_offset + i * window_height:
                              HistogramPeakDetection.settings.lanes.top_offset + (i+1) * window_height + 1, :]
      vertical_window_average = numpy.average(window, axis=0)
      vertical_averages[0, i, :] = vertical_window_average[0: width // HistogramPeakDetection.settings.lanes.num_to_detect]
      vertical_averages[1, i, :] = vertical_window_average[width // HistogramPeakDetection.settings.lanes.num_to_detect : ]

    window_maxes = numpy.iinfo(numpy.uint16).max * numpy.ones((HistogramPeakDetection.settings.lanes.num_to_detect,
                                                               num_vertical_windows, n_max_per_window), numpy.uint16)
    for i in range(num_vertical_windows):
      for lane in range(HistogramPeakDetection.settings.lanes.num_to_detect):
        n_max_indices = numpy.argpartition(vertical_averages[lane, i, :], -n_max_per_window)[-n_max_per_window:]
        maxes_greater_than_threshold_indices = vertical_averages[lane, i, n_max_indices] >= HistogramPeakDetection.settings.sliding_window.active_threshold
        n_max_indices = n_max_indices[maxes_greater_than_threshold_indices]
        window_maxes[lane, i, 0:n_max_indices.shape[0]] = n_max_indices
        # if numpy.max(vertical_averages[lane, i, :]) >= 150:
        #   window_maxes[lane, i] = numpy.argmax(vertical_averages[lane, i, :])

    if self.debug:
      plot.bar(numpy.arange(width), numpy.average(detected_image, axis=0))
      plot.show()
      time.sleep(7.5)

    points: list[list[tuple[int, int]]] = [[] for i in range(HistogramPeakDetection.settings.lanes.num_to_detect)]
    for window_index in range(num_vertical_windows):
      y_cord = HistogramPeakDetection.settings.lanes.top_offset + window_index * window_height + window_height // 2

      for lane in range(HistogramPeakDetection.settings.lanes.num_to_detect):
        for i in range(n_max_per_window):
          if window_maxes[lane, window_index, i] != numpy.iinfo(numpy.uint16).max:
            points[lane].append((window_maxes[lane, window_index, i] + lane * width // HistogramPeakDetection.settings.lanes.num_to_detect,
                                 y_cord))

    left_points, right_points = points
    return numpy.array(left_points), numpy.array(right_points)

  def filter_points(self, points: numpy.array, reject_outside_n_std_devs: float = 1.5, max_iterations: int = 5,
                    max_residual: int = 20) -> numpy.array:

    assert len(points) >= 2
    assert numpy.all(numpy.diff(points[:, 1]) >= 0)  # sorted by y value

    def line_of_best_fit(data: numpy.array) -> tuple[numpy.poly1d, numpy.array, float]:
      assert len(points) > 1  # there are points to fit a line on
      best_fit_coef = numpy.polyfit(data[:, 1], data[:, 0], deg=HistogramPeakDetection.DEGREE_TO_FIT_TO_LINES)
      best_fit = numpy.poly1d(best_fit_coef)

      residuals = numpy.linalg.norm((best_fit(data[:, 1]) - data[:, 0]).reshape(len(data), 1), axis=1)
      residuals_std_dev = numpy.std(residuals)
      mean_centered_residuals = residuals - numpy.mean(residuals)

      return best_fit, mean_centered_residuals, residuals_std_dev

    for iter in range(max_iterations):
      best_fit, mean_centered_residuals, residuals_std_dev = line_of_best_fit(points)
      points_to_keep = mean_centered_residuals <= reject_outside_n_std_devs * residuals_std_dev

      if self.debug:
        plot.bar(numpy.arange(mean_centered_residuals.shape[0]), mean_centered_residuals)
        plot.show()
        time.sleep(7.5)

      if numpy.all(points_to_keep):  # no points to remove, break early
        break
      else:  # we have points to remove -> remove them
        points = points[points_to_keep]

    best_fit, mean_centered_residuals, residuals_std_dev = line_of_best_fit(points)
    points = points[mean_centered_residuals <= max_residual]  # enforce a hard limit on the maximum residual

    return points

  def fit_lane_to_points(self, points, polynomial_degree: int = DEGREE_TO_FIT_TO_LINES)\
      -> tuple[numpy.array, numpy.array]:

    left_points, right_points = points
    left_lane = numpy.polyfit(left_points[:, 1], left_points[:, 0], deg=polynomial_degree)
    right_lane = numpy.polyfit(right_points[:, 1], right_points[:, 0], deg=polynomial_degree)
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
    polylines = numpy.empty((len(lanes), *lin.shape, 2), numpy.int32)
    for lane, lane_poly_coeff in enumerate(lanes):
      lane_poly = numpy.poly1d(lane_poly_coeff)
      for i, val in enumerate(lin):
        polylines[lane, i, :] = numpy.array([lane_poly(val), val], numpy.int32)

    lane_image = numpy.empty_like(image)
    for lane in range(len(lanes)):
      cv2.polylines(lane_image, numpy.int32([polylines[lane, :]]), False, color, thickness)
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
    saturation_filter_mask = self.filter_thresholds(s_channel, (6, 85))
    self._add_knot('Saturation  Filter Mask', saturation_filter_mask, hls=False)
    saturation_filtered = cv2.bitwise_and(s_channel, saturation_filter_mask)
    self._add_knot('Saturation  Filtered', saturation_filtered, hls=False)

    combined_filter = sobel_filter_mask & saturation_filter_mask
    self._add_knot('Combined Filters', combined_filter, hls=False)

    # fill contours in combined filters
    contours = cv2.findContours(combined_filter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for contour in contours:
      cv2.drawContours(combined_filter, [contour], -1, (255, 255, 255), -1)
    self._add_knot('Filled Contours Combined Filters', combined_filter, hls=False)

    masked = region_of_interest(self, combined_filter)
    self._add_knot('Region Of Interest Mask', masked, hls=False)

    # do the 'Histogram Peak Detection'
    left_points, right_points = self.get_histogram_peak_points(masked)
    combined_points = [*left_points, *right_points]

    # display the detected points on the image
    masked_image_rgb = cv2.cvtColor(masked, cv2.COLOR_GRAY2RGB)

    self._display_points(masked_image_rgb, left_points, color=(255, 255, 0), overlay_name='Detected Left Points')
    self._display_points(masked_image_rgb, right_points, color=(0, 255, 255), overlay_name='Detected Right Points')

    points_image = self._display_points(masked_image_rgb, combined_points, display_overlay=False)
    detected_points_result = cv2.addWeighted(masked_image_rgb, 0.5, points_image, 1, 0)
    self._add_knot('Detected Points Result', detected_points_result, hls=False)

    # filter the points to remove 'incorrect' detections
    left_points = self.filter_points(left_points)
    right_points = self.filter_points(right_points)
    combined_points = [*left_points, *right_points]

    left_points_image = self._display_points(masked_image_rgb, left_points, color=(255, 255, 0),
                                             overlay_name='Filtered Left Points')
    right_points_image = self._display_points(masked_image_rgb, right_points, color=(0, 255, 255),
                                              overlay_name='Filtered Right Points')

    points_image = self._display_points(masked_image_rgb, combined_points, display_overlay=False)
    detected_points_result = cv2.addWeighted(masked_image_rgb, 0.5, points_image, 1, 0)
    self._add_knot('Filtered Points Result', detected_points_result, hls=False)

    lanes = self.fit_lane_to_points((left_points, right_points))
    self._display_lanes(frame, (lanes[0],), overlay_name='Left Lane')
    self._display_lanes(frame, (lanes[1],), overlay_name='Right Lane')

    lane_image = self._display_lanes(frame, lanes)
    detected_lanes_result = cv2.addWeighted(frame, 0.75, lane_image, 1, 0)
    detected_lanes_result = cv2.addWeighted(detected_lanes_result, 1, left_points_image, 1, 0)
    detected_lanes_result = cv2.addWeighted(detected_lanes_result, 1, right_points_image, 1, 0)
    self._add_knot('Detected Lanes Result', detected_lanes_result, hls=False)

    lanes = self._historic_fill.get(numpy.array(lanes))
    lane_image = self._display_lanes(frame, lanes, overlay_name='Historic Filtered')
    detected_lanes_result = cv2.addWeighted(frame, 0.75, lane_image, 1, 0)
    self._add_knot('Historic Filtered Result', detected_lanes_result, hls=False)

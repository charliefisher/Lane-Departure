from threading import Thread

import cv2
import numpy
from matplotlib import pyplot as plot
from playsound import playsound

from lane_detection.pipeline import Pipeline
from lane_detection.pipeline.general import display_lines
from general import constants


class LaneDeparture(Thread):
  def __init__(self, lane_detection: Pipeline) -> None:
    super().__init__(name=self.__class__.__name__)
    self._lane_detection = lane_detection
    self.start()

  def departure_error(self, base_image, lines) -> None:
    ### warn in console ###
    print('DEPARTURE WARNING')
    ### audio warning ###
    playsound('./resources/warning.mp3', block=False)
    ### visual warning ###
    # convert lines to polygon inside lines
    left_line, right_line = lines
    l_x1, l_y1, l_x2, l_y2 = left_line.reshape(4)
    r_x1, r_y1, r_x2, r_y2 = right_line.reshape(4)
    polygon = [(l_x1, l_y1), (l_x2, l_y2), (r_x2, r_y2), (r_x1, r_y1)]

    # draw polygon outlining lines
    poly_base = numpy.zeros_like(base_image)  # define a base image to draw the polygon on (simply a black screen)
    polygons = numpy.empty(shape=(1, len(polygon), 2), dtype=numpy.int32)
    polygons[0] = numpy.array(polygon)
    cv2.fillPoly(poly_base, polygons, (0, 0, 255))  # draw the polygons onto the base image
    error_overlay = cv2.addWeighted(base_image, 1, poly_base, 1, 0)
    return error_overlay

  def run(self) -> None:
    cv2.namedWindow(self.name)  # open window to display lanes (and departure warnings)

    last_median = None
    while True:  # do-while loop
      # read lanes and break if value is sentinel
      frame, lanes = self._lane_detection.start_consumption()
      if lanes is constants.SENTINEL:
        break

      # find median of base of triangle formed by lane lines
      triangle_median = numpy.average(lanes, axis=0)[1]

      lane_image, lines = display_lines(frame, lanes, display_overlay=False)
      lane_overlay = cv2.addWeighted(frame, 1, lane_image, 1, 0)

      if last_median is not None:
        scatter = numpy.fabs(triangle_median - last_median)
        if scatter > 25:  # departure error
          lane_overlay = self.departure_error(lane_overlay, lines)

      last_median = triangle_median
      self._lane_detection.end_consumption()

      # show the lanes and departure warnings
      cv2.imshow(self.name, lane_overlay)
      keypress = cv2.waitKey(1) & 0xFF  # get the keypress
      if keypress == ord('q'):
        cv2.destroyAllWindows()
        break

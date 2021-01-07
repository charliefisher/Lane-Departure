from threading import Thread

import numpy
from matplotlib import pyplot as plot
from playsound import playsound

from lane_detection.pipeline import Pipeline
from general import constants


class LaneDeparture(Thread):
  def __init__(self, lane_detection: Pipeline) -> None:
    super().__init__()
    self._lane_detection = lane_detection
    self.start()

  def departure_error(self) -> None:
    print('DEPARTURE WARNING')
    playsound('./resources/warning.mp3', block=False)

  def run(self) -> None:

    last_median =  None
    scatters = numpy.empty((0,), numpy.float)

    while True:  # do-while loop
      # read lanes and break if value is sentinel
      lanes = self._lane_detection.start_read_lanes()
      if lanes is constants.SENTINEL:
        break

      # find median of base of triangle formed by lane lines
      triangle_median = numpy.average(lanes, axis=0)[1]

      if last_median is not None:
        scatter = numpy.fabs(triangle_median - last_median)

        if scatter > 25:
          self.departure_error()

        scatters = numpy.append(scatters, scatter)

      last_median = triangle_median

      self._lane_detection.finish_read_lanes()

    plot.bar(numpy.arange(len(scatters)), scatters)
    plot.show()
    import time
    time.sleep(7.5)
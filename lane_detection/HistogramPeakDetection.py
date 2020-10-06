import numpy

from lane_detection.Pipeline import Pipeline


class HistogramPeakDetection(Pipeline):
  """
   Explain Hough Transform here
   """

  DEGREE_TO_FIT_TO_LINES = 5

  def __init__(self, source, should_start, show_pipeline, debug):
    """
    Calls superclass __init__ (see Pipeline.__init__ for more details)

    :param source: the filename or device that the pipeline should be run on
    :param should_start: a flag indicating whether or not the pipeline should start as soon as it is instantiated
    :param show_pipeline: a flag indicating whether or not each step in the pipeline should be shown
    :param debug: a flag indicating whether or not the use is debugging the pipeline. In debug, the pipeline is
                  shown and debug statements are enabled
    """

    super().__init__(source, should_start, show_pipeline, debug, True)

  def _run(self, frame):
    """
    Histogram Peak Detection is run on the frame to detect the lane lines

    This method handles detecting the lines, drawing them, and passing them to the lane departure algorithm

    *insert more info about Histogram Peak Detection here*

    :param frame: the current frame of the capture
    :return: void
    """

    frame = numpy.copy(frame)
    self._add_knot('Raw', frame)

    if self._show_pipeline:
      self._display_pipeline()
from pipelines.Pipeline import Pipeline

import cv2
# import matplotlib.pyplot as plot


class HoughTransform(Pipeline):
  """
  Explain Hough Transform here
  """

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

  def _run(self, frame):
    """
    Hough Transform is run on the frame to detect the lane lines

    This method handles detecting the lines, drawing them, and passing them to the lane departure algorithm

    *insert more info about Hough Transform here*

    :param frame: the current frame of the capture
    :return: void
    """
    return

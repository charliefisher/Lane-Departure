from abc import ABC, abstractmethod
from multiprocessing import Process
import time
import cv2


class Pipeline(ABC):
  """
  A superclass for OpenCV lane detection pipelines

  It handles opening and closing a video feed and subclasses simply implement the lane detection algorithm in their
  _run() method.

  Use:
    start() to open video feed and calls run()
    _run() is where the lane detection algorithm is implemented (MUST be overriden by subclass)
    stop() to close video feed and windows and stop calling run()

  :ivar _show_pipeline: a flag indicating whether or not each step in the pipeline should be shown
  :ivar _debug: a flag indicating whether or not the use is debugging the pipeline. In debug, the pipeline is shown and
               debug statements are enabled
  :ivar _capture: the OpenCV capture object (CvCapture) that the lane detection algorithm should run on
  """

  def __init__(self, source, should_start, show_pipeline, debug):
    """
    Declares instance variables (_show_pipeline, _debug, _capture) and starts the pipeline according to should_start

    :param source: the filename or device that the pipeline should be run on
    :param should_start: a flag indicating whether or not the pipeline should start as soon as it is instantiated
    :param show_pipeline: a flag indicating whether or not each step in the pipeline should be shown
    :param debug: a flag indicating whether or not the use is debugging the pipeline. In debug, the pipeline is
                  shown and debug statements are enabled
    """
    self.__stop = False
    self.__source = source
    self._show_pipeline = show_pipeline or debug
    self._debug = debug
    self._capture = None

    self.__process = Process(target=self._run_pipeline, args=(self,))
    self.__process.daemon = False

    # check if the pipeline should start immediately
    if should_start:
      self.start()

  def start(self):
    """
    Opens the video feed and handles the looping of run()

    :return: void
    """

    self.__process.start()

  @classmethod
  def __open_source(self, capture, input):
    """
    Opens a cv2.capture object

    This method is susceptible to raise any errors caused by cv2.VideoCapture(src)

    :param input: the filename or device id to be opened
    :raises: Exception is raised if this method is called when _capture is already open
    :return: void
    """

    if not capture:
      capture = cv2.VideoCapture(input)
    else:
      raise Exception('Cannot open {input} as a capture is already open'.format(input=input))
    return capture

  @classmethod
  def _run_pipeline(self, parent):
    """

    :return: void
    """

    # open input
    parent._capture = self.__open_source(parent._capture, parent.__source)
    # get fps of video
    fps = parent._capture.get(cv2.CAP_PROP_FPS)
    # loop run() while the capture is open and we we have not stopped running
    while not parent.__stop and parent._capture.isOpened():
      # store start time of loop
      start_time = time.time()
      # read a frame of the capture
      return_value, frame = parent._capture.read()
      # check that the next frame was read successfully
      # i.e. that we have not hit the end of the video or encountered an error
      if return_value:
        self._run(self, frame)
        # check if 'q' is pressed to stop pipeline
        if cv2.waitKey(1) & 0xFF == ord('q'):
          self.stop(parent)
      else:
        # stop the pipeline if we hit the end of the video or encountered an error
        self.stop(parent)
      # only sleep if stop was not called (i.e. we will read the next frame)
      if not parent.__stop:
        # 1 second / fps = time to sleep for each frame subtract elapsed time
        time_to_sleep = max(1 / fps - (time.time() - start_time), 0)
        time.sleep(time_to_sleep)

  @classmethod
  def stop(self, parent):
    """
    Closes _capture and all windows and stops looping run()

    :return: void
    """

    parent._capture.release()
    cv2.destroyAllWindows()
    parent.__stop = True
    # do not have to worry about joining the thread on stop since it is not daemonic and non daemonic threads are joinged automatically

  @classmethod
  @abstractmethod
  def _run(self, frame):
    """
    @Override - subclass MUST override this function
    Where the lane detection algorithm is written, it is called on each frame of _capture.

    :param frame: the frame of the capture that the pipeline should be run on
    :return: void
    """
    return

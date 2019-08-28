from abc import ABC, abstractmethod
from multiprocessing import Process
import time
import cv2


class Pipeline(ABC, Process):
  """
  A superclass for OpenCV lane detection pipelines

  It handles opening and closing a video feed and subclasses simply implement the lane detection algorithm in their
  _run() method.

  Use:
    start() to open video feed and calls run()
    _run() is where the lane detection algorithm is implemented (MUST be overriden by subclass)
    stop() to close video feed and windows and stop calling run()

  :ivar _pipeline: a list of frames showing the different steps of the pipeline. It should only ever store the pipeline
                for a single iteration at a given instant (i.e. its length should never exceed the number of steps in
                the pipeline)
  :ivar _show_pipeline: a flag indicating whether or not each step in the pipeline should be shown
  :ivar _debug: a flag indicating whether or not the use is debugging the pipeline. In debug, the pipeline is shown and
               debug statements are enabled
  :ivar _capture: the OpenCV capture object (CvCapture) that the lane detection algorithm should run on
  """

  def __init__(self, source, should_start, show_pipeline, debug):
    """
    Declares instance variables (_show_pipeline, _debug, _capture) and starts the pipeline according to should_start

    :param name: the name of the pipeline (the name of the subclass)
    :param source: the filename or device that the pipeline should be run on
    :param should_start: a flag indicating whether or not the pipeline should start as soon as it is instantiated
    :param show_pipeline: a flag indicating whether or not each step in the pipeline should be shown
    :param debug: a flag indicating whether or not the use is debugging the pipeline. In debug, the pipeline is
                  shown and debug statements are enabled
    """

    # call superclass constructor
    super().__init__()
    # initialize instance variables
    self.__stop = False
    self.__source = source
    class_name = str(self.__class__)
    self._name = class_name[class_name.rindex('.') + 1:-2]
    self._pipeline = []
    self._show_pipeline = show_pipeline or debug
    self._debug = debug
    self._capture = None

    # check if the pipeline should start immediately
    if should_start and not self.is_alive():
      self.start()

  def start(self):
    """
    Starts running the process which then subsequently opens the video and runs the pipeline

    :return: void
    """

    # call Process.start() to start the Process execution
    super().start()

  def __open_source(self, input):
    """
    Opens a cv2.capture object

    This method is susceptible to raise any errors caused by cv2.VideoCapture(src)

    :param input: the filename or device id to be opened
    :raises: Exception is raised if this method is called when _capture is already open
    :return: void
    """

    # check that capture is not already open
    if not self._capture:
      # open capture from provided input
      self._capture = cv2.VideoCapture(input)
    else:
      # throw error if capture is already open
      raise Exception('Cannot open {input} as a capture is already open'.format(input=input))

  def run(self):
    """

    :return: void
    """

    # open input
    self.__open_source(self.__source)
    # get fps of video
    fps = self._capture.get(cv2.CAP_PROP_FPS)
    # loop run() while the capture is open and we we have not stopped running
    while not self.__stop and self._capture.isOpened():
      # store start time of loop
      start_time = time.time()
      # read a frame of the capture
      return_value, frame = self._capture.read()
      # check that the next frame was read successfully
      # i.e. that we have not hit the end of the video or encountered an error
      if return_value:
        self._run(frame)
        # check if 'q' is pressed to stop pipeline
        if cv2.waitKey(1) & 0xFF == ord('q'):
          self.stop()
      else:
        # stop the pipeline if we hit the end of the video or encountered an error
        self.stop()
      # only sleep if stop was not called (i.e. we will read the next frame)
      if not self.__stop:
        # 1 second / fps = time to sleep for each frame subtract elapsed time
        time_to_sleep = max(1 / fps - (time.time() - start_time), 0)
        time.sleep(time_to_sleep)

  def stop(self):
    """
    Closes _capture and all windows and stops looping run()

    :return: void
    """

    # close the capture
    self._capture.release()
    # remove all windows
    cv2.destroyAllWindows()
    # set flag to stop process execution
    self.__stop = True
    # do not have to worry about joining the thread on stop since it is not daemonic and non daemonic threads are joinged automatically

  def _add_knot(self, name, image):
    """
    Adds a knot to the lane detection pipeline

    :param name: the name of the image to be added
    :param image: the image to be added to the end of the pipeline
    :return: void
    """
    self._pipeline.append((name, image))

  def _clear_pipeline(self):
    """
    Empties the stored steps of the lane detection pipeline

    :return: void
    """
    self._pipeline = []

  def _display_pipeline(self):
    """
    Displays the pipeline to the user. Depending on the state of _show_pipeline, the steps of the pipeline may be shown

    :return: void
    """

    def display_knot(knot, index):
      """
      Displays a single knot in the pipeline

      :param knot: the knot in the pipeline to be displayed
      :param index: the index of the knot in the pipeline
      :return: void
      """
      # destructure the knot
      name, image = knot
      # display the knot with the correct name and title
      cv2.imshow('[{} - {}]  -  {}'.format(self._name, index, name), image)

    # display the steps of the pipeline only if that option is selected
    if self._show_pipeline:
      # iterate through all but the final step and display those knots in the pipeline
      for i in range(len(self._pipeline[:-1])):
        display_knot(self._pipeline[i], i+1)
    # display the result (regardless of whether or not the pipeline should be shown)
    display_knot(self._pipeline[-1], len(self._pipeline))
    # reset the pipeline now that it has been displayed
    self._clear_pipeline()

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

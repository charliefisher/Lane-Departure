from configparser import ConfigParser
from os import path
from abc import ABC, abstractmethod
from multiprocessing import Process
import math
import re
import time
import cv2
import numpy


class Pipeline(ABC, Process):
  """
  A superclass for OpenCV lane detection

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

  # set up config file reader
  __config = ConfigParser(allow_no_value=True)
  __config.read(path.join(path.dirname(__file__), r'./Pipeline.config'))
  # set up static variables from config file
  SCREEN_WIDTH = int(__config['window']['width'])
  SCREEN_HEIGHT = int(__config['window']['height'])
  FINAL_IMAGE_RATIO = float(__config['display']['final_image_ratio'])
  FONT_FACE = vars(cv2)[__config['font']['font_face']]
  FONT_COLOR = tuple(map(int, re.sub('\(|\)| ', '', __config['font']['color']).split(',')))
  FONT_THICKNESS = int(__config['font']['thickness'])
  FONT_SCALE = float(__config['font']['scale'])
  FONT_EDGE_OFFSET = int(__config['font']['edge_offset'])
  NUM_IMAGE_CHANNELS = 3


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
    self.__show_pipeline_steps = True
    class_name = str(self.__class__)
    self._name = class_name[class_name.rindex('.') + 1:-2]
    self._screen = numpy.zeros((Pipeline.SCREEN_HEIGHT, Pipeline.SCREEN_WIDTH, Pipeline.NUM_IMAGE_CHANNELS), numpy.uint8)
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
    Displays the pipeline to the user. Depending on the state of __show_pipeline_steps, the steps of the pipeline may
    or may not be visible.

    :return: void
    """

    def add_knot_to_screen(index, knot, new_dimension, position):
      """
      Displays a single knot in the pipeline

      :param index: the index of the knot in the pipeline
      :param knot: the knot in the pipeline to be displayed
      :param new_dimension: the desired size of the knot to be displayed - a tuple of the form (width, height)
      :param position: the position of the top left corner of the image on self._screen - a tuple of the form (y, x)
      :return: void
      """

      # destructure the knot
      name, image = knot
      # resize the image to the desired size
      resized_image = cv2.resize(image, dsize=new_dimension)
      # add the image to the screen at the specified location
      start_y, start_x = position
      width, height = new_dimension
      self._screen[start_y:(start_y + height), start_x:(start_x + width)] = resized_image

      # add the title of the knot to the image
      title = '{}  -  {}'.format(index, name)
      title_bounding_box, title_basline = cv2.getTextSize(title, Pipeline.FONT_FACE, Pipeline.FONT_SCALE, Pipeline.FONT_THICKNESS)
      text_width, text_height = title_bounding_box
      position = (start_x + Pipeline.FONT_EDGE_OFFSET, start_y + text_height + Pipeline.FONT_EDGE_OFFSET)
      cv2.putText(self._screen, title, position, Pipeline.FONT_FACE, Pipeline.FONT_SCALE, Pipeline.FONT_COLOR, Pipeline.FONT_THICKNESS)

    pipeline_steps = self._pipeline[:-1]
    final_step = self._pipeline[-1]
    num_pipeline_steps = len(pipeline_steps)

    # display the steps of the pipeline only if that option is selected
    if self.__show_pipeline_steps and num_pipeline_steps > 0:
      aspect_ratio = None
      # check that all steps of the pipeline have the same aspect ratio (if not raise and error)
      # simultaneously, check if any images are single channel and convert them to the correct number of channels
      for i in range(len(self._pipeline)):
        name, image = self._pipeline[i]
        # get the dimensions of the image
        # note that if the image is single channel, then num_channels will be undefined -> set it to default value after
        height, width, *num_channels = image.shape
        num_channels = num_channels[0] if num_channels else 1

        # check for aspect ratio consistency throughout the pipeline
        if aspect_ratio is None:
          aspect_ratio = height / width
        elif height / width != aspect_ratio:
          raise Exception('aspect ratio of images is not consistent throughout pipeline')

        # if the image is single channel (grayscale), convert it to 3 channels (still grayscale)
        # this allows the images to be merged into one
        if num_channels == 1:
          temp = numpy.empty((height, width, Pipeline.NUM_IMAGE_CHANNELS))
          for channel in range(Pipeline.NUM_IMAGE_CHANNELS):
            temp[:, :, channel] = image
          image = temp
          if i < num_pipeline_steps:
            pipeline_steps[i] = (name, image)
          else:
            final_step = (name, image)

      # return the next lowest square greater than num
      next_square = lambda num: int(round(math.pow(math.ceil(math.sqrt(abs(num))), 2)))

      # num_bins_per_quarter = next_square(math.ceil(num_pipeline_steps / 2))
      num_bins_per_quarter = next_square(math.ceil(num_pipeline_steps * Pipeline.FINAL_IMAGE_RATIO))
      horizontal_bins_dimension = int(round(math.sqrt(num_bins_per_quarter)))
      vertical_bins_dimension = 2 * horizontal_bins_dimension

      # if int(1 / Pipeline.FINAL_IMAGE_RATIO) != int(round(1 / Pipeline.FINAL_IMAGE_RATIO)):
      #   raise Exception("This ratio doesn't work boiii")
      #
      # vertical_bins_dimension = (1 / Pipeline.FINAL_IMAGE_RATIO) * horizontal_bins_dimension

      container_width = int(round(Pipeline.SCREEN_WIDTH * (1 - Pipeline.FINAL_IMAGE_RATIO)))

      step_width = container_width // horizontal_bins_dimension
      step_height = int(round(step_width * aspect_ratio))

      print('bins are XxY', horizontal_bins_dimension, 'x', vertical_bins_dimension)
      print('step is XxY', step_width, 'x', step_height)

      # iterate through all but the final step and display those knots in the pipeline
      i = 0
      for name, image in pipeline_steps:
        # add the knot to the screen at the correct position
        # start_y = step_height * (i // int(round(math.ceil(num_pipeline_steps / vertical_bins_dimension))))
        start_y = step_height * (i // horizontal_bins_dimension)
        start_x = step_width * (i % horizontal_bins_dimension)
        print('adding knot at (', start_x, ', ', start_y, ')')
        add_knot_to_screen(i + 1, knot=(name, image), new_dimension=(step_width, step_height), position=(start_y, start_x))

        i += 1

      # add the final step to the screen in the bottom left quarter
      output_width = int(round(Pipeline.SCREEN_WIDTH * Pipeline.FINAL_IMAGE_RATIO))
      output_height = int(round(Pipeline.SCREEN_HEIGHT * Pipeline.FINAL_IMAGE_RATIO))
      add_knot_to_screen(len(self._pipeline), knot=final_step, new_dimension=(output_width, output_height), position=(Pipeline.SCREEN_HEIGHT-output_height, Pipeline.SCREEN_WIDTH-output_width))

      cv2.imshow(self._name, self._screen)
    else:
      name, image = final_step
      cv2.imshow(self._name, image)
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

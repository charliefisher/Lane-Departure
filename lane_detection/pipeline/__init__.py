import os
import inspect
import math
import time
import cv2
import numpy
from threading import Thread
from multiprocessing import Process
from abc import ABC, abstractmethod
from datetime import datetime

import settings as _settings
import pipeline.utils as utils
from general.friend import Friendable, register_friend

settings = _settings.load(_settings.SettingsCategories.PIPELINES, _settings.PipelineSettings.GENERAL)

NUM_IMAGE_CHANNELS = 3




@register_friend(utils.Visualizer)
class Pipeline(ABC, Process, Friendable):
  """
  A superclass for OpenCV lane detection

  It handles opening and closing a video feed and subclasses simply implement the lane detection algorithm in their
  _run() method. Additionally, provides tools to visualize the steps of a lane detection pipeline and for the user to
  manually apply a mask to an image.

  Use:
    start() to open video feed and calls _run()
    _run() is where the lane detection algorithm is implemented (MUST be overriden by subclass)
    stop() to close video feed and windows and stop calling run()

    #TODO: update this list
    take_screenshot()
    _add_knot()


  # TODO: update this list
  :ivar _pipeline: a list of frames showing the different steps of the pipeline. It should only ever store the pipeline
                for a single iteration at a given instant (i.e. its length should never exceed the number of steps in
                the pipeline) - not guaranteed to be filled
  :ivar _show_pipeline: a flag indicating whether or not each step in the pipeline should be shown
  :ivar _debug: a flag indicating whether or not the use is debugging the pipeline. In debug, the pipeline is shown and
                debug statements are enabled
  :ivar _capture: the OpenCV capture object (CvCapture) that the lane detection algorithm should run on
  :ivar _fps: the fps of the capture object that the lane detection algorithmn is run on
  :ivar _name: the name of the pipeline (derived from the class name)
  :ivar _screen: the image where the pipeline steps are drawn
  :ivar _image_mask_enabled: indicates whether or not the current instance of the pipeline supports image masks
  :ivar _region_of_interest_mask: stores the region of interest mask (will be empty if image mask is disabled)

  :ivar __stop: indicates whether or not the pipeline is stopped
  :ivar __source: the source that the pipeline is being run on
  :ivar __show_pipeline_steps: indicates whether or not the intermediate steps in the pipeline should be shown or just
                the final step
  :ivar __cached_pipeline: stores the last snapshot of the pipeline since it was cleared (since the current one is not
                guaranteed to be filled)
  :ivar __paused: indicates whether or not the pipeline is currently paused
  :ivar __while_paused: stores the function to be executed while the pipeline is paused

  :friend pipeline.utils.Visualizer
  """

  def __init__(self, source: str, *,
               should_start: bool = True,
               show_pipeline: bool = True,
               image_mask_enabled: bool = True,
               debug: bool = False):
    """
    Declares instance variables (_show_pipeline, _debug, _capture) and starts the pipeline according to should_start

    :param source: the filename or device that the pipeline should be run on
    :param should_start: a flag indicating whether or not the pipeline should start as soon as it is instantiated
    :param show_pipeline: a flag indicating whether or not each step in the pipeline should be shown
    :param debug: a flag indicating whether or not the use is debugging the pipeline. In debug, the pipeline is
                  shown and debug statements are enabled
    """

    # call superclass constructor
    super().__init__()

    # initialize instance variables

    # private - use property accessor
    self._source = source
    self._frame = None
    self._name = self.__class__.__name__
    self._fps = None
    self._image_mask_enabled = image_mask_enabled
    self._debug = debug
    self._show_pipeline = show_pipeline or self._debug
    self._show_pipeline_steps = settings.display.show_pipeline_steps
    self._knots = []

    screen_dimensions = (settings.window.height, settings.window.width, NUM_IMAGE_CHANNELS)
    # protected
    self._screen = numpy.zeros(screen_dimensions, numpy.uint8)
    self._visualizer = None
    self._region_of_interest = None
    self._capture = None

    # private - only accessible by class
    self.__current_knots = []  # maybe not be filled (most likely, will be partially filled)
    self.__stop = False
    self.__paused = False
    self.__while_paused = None

    # check if the pipeline should start immediately
    if should_start and not self.is_alive():
      self.start()

##### Property Accessors for Read-Only Instance Variables #####
  @property
  def source(self) -> str:
    return self._source

  @property
  def frame(self) -> numpy.array:
    return self._frame

  @property
  def name(self) -> str:
    return self._name

  @property
  def fps(self) -> int:
    return self._fps

  @property
  def image_mask_enabled(self) -> bool:
    return self._image_mask_enabled

  @property
  def knots(self) -> list[numpy.array]:
    return self.__current_knots

  @property
  def region_of_interest(self) -> list[tuple[int, int]]:
    return self._region_of_interest.get()

##### Property Accessors and Mutators for Instance Variables #####
  @property
  def show_pipeline(self) -> bool:
    return self._show_pipeline

  @show_pipeline.setter
  def show_pipeline(self, value: bool) -> None:
    self._show_pipeline = value

  @property
  def show_pipeline_steps(self) -> bool:
    return self._show_pipeline_steps

  @show_pipeline_steps.setter
  def show_pipeline(self, value: bool) -> None:
    self._show_pipeline_steps = value

  @property
  def debug(self) -> bool:
    return self._debug

  @debug.setter
  def debug(self, value: bool) -> None:
    self._debug = value

##### Method Definitions #####
  def start(self):
    """
    Starts running the process which then subsequently opens the video and runs the pipeline

    :return: void
    """

    super().start()  # call Process.start() to start the Process execution

  def __open_source(self, src: str) -> None:
    """
    Opens a cv2.capture object

    This method is susceptible to raise any errors caused by cv2.VideoCapture(src)

    :param input: the filename or device id to be opened
    :raises: RuntimeError is raised if this method is called when _capture is already open
    :return: void
    """

    # check that capture is not already open
    if not self._capture:
      self._capture = cv2.VideoCapture(src)  # open capture from provided input
      self._fps = self._capture.get(cv2.CAP_PROP_FPS)  # get fps of video
    else:
      # throw error if capture is already open
      raise RuntimeError('Cannot open {input} as a capture is already open'.format(input=input))

  def run(self):
    """

    :return: void
    """

    # check if function was called by Process superclass and raise an error if it was not
    if inspect.stack()[1].function != '_bootstrap':
      raise RuntimeError('pipeline::run can only be invoked by multiprocessing::Process')

    self.__open_source(self._source)  # open input
    first_frame = True  # initialize a flag used to indicate if init_pipeline should be run

    # loop run() while the capture is open and we we have not stopped running
    while not self.__stop and self._capture.isOpened():
      # if the pipeline is not paused, read a frame from the capture and call the pipeline
      if not self.is_paused():
        start_time = time.time()  # store start time of loop
        return_value, frame = self._capture.read()  # read a frame of the capture

        # check that the next frame was read successfully
        # i.e. that we have not hit the end of the video or encountered an error
        if return_value:
          self._frame = frame
          # if it is the first frame of the pipeline, run init_pipeline, then set flag to false
          if first_frame:
            self._init_pipeline(self._frame)
            first_frame = False
          self._run(self._frame)  # run the pipeline
          if self._show_pipeline:  # display the pipeline
            # self.__display_pipeline()
            self._visualizer.get()
        else:
          self.stop()  # stop the pipeline if we hit the end of the video or encountered an error

        # only sleep if stop was not called (i.e. we will read the next frame)
        if not self.__stop:
          # 1 second / fps = time to sleep for each frame subtract elapsed time
          time_to_sleep = max(1 / self._fps - (time.time() - start_time), 0)
          time.sleep(time_to_sleep)
      # if the pipeline is paused and the whilepaused handler is defined, call it
      elif self.__while_paused is not None:
        self.__while_paused()  # NOTE: the pipeline will block until the function returns

      keypress = cv2.waitKey(1) & 0xFF  # get the keypress
      # if a key was pressed, call the handler with the pressed key
      if keypress:
        self.__handle_keypress(keypress)

      # reset the pipeline now that the current iteration has finished
      self.__clear_pipeline()

  def __handle_keypress(self, keypress):
    """
    Handles actions based on a keypress. Works as a delegator to delegate actions based on specific keypress. If the
    keypress maps to a default action, that actions is invoked, otherwise the keypress is passed to the subclass.

    :param keypress: the code of the keypress
    :param frame: the current frame of the pipeline when the keypress occurred
    :return: void
    """

    # q - stop the pipeline
    if keypress == ord('q'):
      self.stop()
    # p - toggle displaying the pipeline steps
    elif keypress == ord('p'):
      self._show_pipeline_steps = not self._show_pipeline_steps
    # s - take a screenshot of the pipeline (saves each knot in the pipeline)
    elif keypress == ord('s'):
      self.take_screenshot()
    # m - allow user to edit mask of source image (if image mask is enabled)
    elif keypress == ord('m') and self._image_mask_enabled:
      self._region_of_interest.editor(self.frame)
    # other non-default case (let the subclass handle these cases if desired)
    else:
      self._handle_keypress(keypress)

  def _handle_keypress(self, keypress):
    """
    @Override - subclass CAN override this function (it is optional)
    Where subclass can add custom keypress events. Cannot override keypress events in pipeline.py. This inhibits the use
    of the 'q', 'p', and 's' keys and possibly the 'm' key, depending on the state of _image_mask_enabled.

    :param keypress: the code of the keypress (will never correspond to any of the keys used for default keypress events)
    :param frame: the current frame of the pipeline when the keypress occurred
    :return: void
    """

    pass


  def _init_pipeline(self, first_frame):
    """
    @Override - subclass CAN override this function (it is optional)
    Where subclass can do any required initialization prior to the pipeline beginning to run. Default action is to get
    the user to apply a mask to the image (if image mask is enabled, otherwise there is no default action).

    :param first_frame: the first frame of the pipeline
    :return: void
    """

    # check if image mask is enabled, if so check if a mask was already defined or get the user to define one
    if self._image_mask_enabled:
      self._region_of_interest = utils.RegionOfInterest(self)
      assert self._region_of_interest.load()  # assert that we loaded a region of interest mask
      self._visualizer = utils.Visualizer(self)

  def is_paused(self):
    """
    Gets whether or not the pipeline is paused

    :return: boolean
    """

    return self.__paused

  def pause(self, whilepaused=None):
    """
    Pauses the pipeline and will call the whilepaused handler until the pipeline is unpaused

    :raises: RuntimeError is raised if this method is called when the pipeline is already paused
    :return: void
    """

    if self.is_paused():
      raise RuntimeError('Cannot pause a pipeline that is already paused')
    self.__paused = True
    self.__while_paused = whilepaused

  def unpause(self):
    """
    Unpauses the pipeline

    :raises: RuntimeError is raised if this method is called when the pipeline is not paused
    :return: void
    """

    if not self.is_paused():
      raise RuntimeError('Cannot unpause a pipeline that is not currently paused')
    self.__paused = False
    self.__while_paused = None

  def stop(self):
    """
    Closes _capture and all windows and stops looping run()

    :return: void
    """

    self._capture.release()  # close the capture
    cv2.destroyAllWindows()  # remove all windows
    self.__stop = True  # set flag to stop process execution

  def take_screenshot(self, extension='jpg'):
    """
    Takes a screenshot of the pipeline (saves each knot in the pipeline)

    :param extension (optional) (default=jpg): the file extension that images should be saved as
    :return: void
    """

    def do_screenshot() -> None:
      output_dir = '{base_output_dir}/{pipeline_name}/{timestamp}' \
        .format(base_output_dir='output',
                pipeline_name=self._name,
                timestamp=datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

      os.makedirs(output_dir, exist_ok=True)

      # iterate through each step of the pipeline and save the corresponding image (which is done in a new thread)
      i = 1
      for name, image in self._knots:
        # format the file name
        file_name = '{index} - {name}.{ext}'.format(index=i, name=name, ext=extension)
        cv2.imwrite(os.path.join(output_dir, file_name), image)
        i += 1

    screenshot = Thread(target=do_screenshot)
    screenshot.start()
    # do not join the thread here since we want the screenshot to occur in the background

  def _add_knot(self, name, image):
    """
    Adds a knot to the lane detection pipeline

    :param name: the name of the image to be added
    :param image: the image to be added to the end of the pipeline
    :return: void
    """
    self.__current_knots.append((name, image))

  def __clear_pipeline(self):
    """
    Empties the stored steps of the lane detection pipeline

    :return: void
    """
    self._knots = self.__current_knots
    self.__current_knots = []

  def __display_pipeline(self):
    """
    Displays the pipeline to the user. Depending on the state of _show_pipeline_steps, the steps of the pipeline may
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
      title = '{index}  -  {name}'.format(index=index, name=name)
      title_bounding_box, title_basline = cv2.getTextSize(title, settings.font.face, settings.font.scale, settings.font.thickness)
      text_width, text_height = title_bounding_box
      position = (start_x + settings.font.edge_offset, start_y + text_height + settings.font.edge_offset)
      cv2.putText(self._screen, title, position, settings.font.face, settings.font.scale, settings.font.color, settings.font.thickness)

    # split the pipeline into the final and intermediate steps
    pipeline_steps = self.__current_knots[:-1]
    final_step = self.__current_knots[-1]
    num_pipeline_steps = len(pipeline_steps)

    # display the steps of the pipeline only if that option is selected
    if self._show_pipeline_steps and num_pipeline_steps > 0:
      # initialize the aspect ratio (gets set later when the pipeline is checked for consistent aspect ratios)
      aspect_ratio = None
      # check that all steps of the pipeline have the same aspect ratio (if not raise and error)
      # simultaneously, check if any images are single channel and convert them to the correct number of channels
      for i in range(len(self.__current_knots)):
        name, image = self.__current_knots[i]
        # get the dimensions of the image
        # note that if the image is single channel, then num_channels will be undefined -> set it to default value after
        height, width, *num_channels = image.shape
        num_channels = num_channels[0] if num_channels else 1

        # check for aspect ratio consistency throughout the pipeline
        if aspect_ratio is None:
          aspect_ratio = height / width
        elif height / width != aspect_ratio:
          raise RuntimeError('aspect ratio of images is not consistent throughout pipeline')

        # if the image is single channel (grayscale), convert it to 3 channels (still grayscale)
        # this allows the images to be merged into one
        if num_channels == 1:
          temp_image = numpy.empty((height, width, Pipeline.NUM_IMAGE_CHANNELS))
          for channel in range(Pipeline.NUM_IMAGE_CHANNELS):
            temp_image[:, :, channel] = image
          if i < num_pipeline_steps:
            pipeline_steps[i] = (name, temp_image)
          else:
            final_step = (name, temp_image)

      # return the next lowest square greater than num
      next_square = lambda num: int(round(math.pow(math.ceil(math.sqrt(abs(num))), 2)))

      # the actual ratio of the final image (will be grater than or equal to settings.display.minimum_final_image_ratio)
      RESULT_IMAGE_RATIO = settings.display.minimum_final_image_ratio
      # initialize variables concerned with the size of pipeline step bins
      # (will get set later when calculating RESULT_IMAGE_RATIO)
      num_bins_top_left = None
      horizontal_bins_dimension = None
      vertical_bins_dimension = None

      # minimic a do-while loop
      while True:
        def calculate_dimensions_given_ratio(ratio):
          """
          Calculates pipeline step bin dimensions given a ratio for the final step of the pipeline

          :param ratio: the ratio of the final step of the pipeline to the rest of the screen
          :return: void
          """

          # allow this function to modify specific variables in outer scope
          nonlocal num_bins_top_left, horizontal_bins_dimension, vertical_bins_dimension
          # do the bin calculations
          num_bins_top_left = next_square(math.ceil(num_pipeline_steps * (1 - ratio)))
          horizontal_bins_dimension = int(round(math.sqrt(num_bins_top_left)))
          vertical_bins_dimension = math.pow(1 - ratio, -1) * horizontal_bins_dimension

        # calculate the bin dimensions for the current ratio
        calculate_dimensions_given_ratio(RESULT_IMAGE_RATIO)
        # if the number of vertical bins is an integer (divides evenly into the screen), then break the loop
        # (the while condition of the do-while loop)
        if vertical_bins_dimension.is_integer():
          break
        # store the previously calculated ratio
        prev = RESULT_IMAGE_RATIO
        # calculate the new ratio to use
        RESULT_IMAGE_RATIO = 1 - horizontal_bins_dimension / math.ceil(vertical_bins_dimension)
        # due to floating point precision errors, sometimes repeating decimals get rounded in an undesirable manner
        # essentially, the program has successfully found the desired ratio, but rounds it causing the program to fail
        # if this occurs, raise an error and instruct the user to fix the rounding error and update the value in
        # pipeline settings
        if prev == RESULT_IMAGE_RATIO:
          raise FloatingPointError('Failed trying to find best ratio for result image. This was caused by a floating point decimal error on repeating digits. Update the pipeline.config file and try again. The recomended ratio is {new_ratio} (simply fix the repeating decimals)'.format(new_ratio=RESULT_IMAGE_RATIO))

      # calculate the dimensions of a pipeline step
      container_width = int(round(settings.window.width * (1 - RESULT_IMAGE_RATIO)))
      step_width = container_width // horizontal_bins_dimension
      step_height = int(round(step_width * aspect_ratio))

      # iterate through all but the final step and display those knots in the pipeline
      i = 0
      for name, image in pipeline_steps:
        # add the knot to the screen at the correct position
        start_y = step_height * (i // horizontal_bins_dimension)
        start_x = step_width * (i % horizontal_bins_dimension)
        add_knot_to_screen(i + 1, knot=(name, image), new_dimension=(step_width, step_height), position=(start_y, start_x))

        i += 1

      # add the final step to the screen in the bottom left quarter
      output_width = int(round(settings.window.width * RESULT_IMAGE_RATIO))
      output_height = int(round(settings.window.height * RESULT_IMAGE_RATIO))
      add_knot_to_screen(len(self.__current_knots), knot=final_step, new_dimension=(output_width, output_height), position=(settings.window.height-output_height, settings.window.width-output_width))

      cv2.imshow(self._name, self._screen)
    else:
      name, image = final_step
      cv2.imshow(self._name, image)


  @abstractmethod
  def _run(self, frame):
    """
    @Override - subclass MUST override this function
    Where the lane detection algorithm is written, it is called on each frame of _capture.

    :param frame: the frame of the capture that the pipeline should be run on
    :return: void
    """

    pass

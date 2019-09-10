from configparser import ConfigParser
from os import path, mkdir
from datetime import datetime
from threading import Thread
from abc import ABC, abstractmethod
from multiprocessing import Process
import inspect
import math
import re
import time
import cv2
import numpy


class Pipeline(ABC, Process):
  """
  A superclass for OpenCV lane detection

  It handles opening and closing a video feed and subclasses simply implement the lane detection algorithm in their
  _run() method. Additionally, provides tools to visualize the steps of a lane detection pipeline and for the user to
  manually apply a mask to an image.

  Use:
    start() to open video feed and calls run()
    _run() is where the lane detection algorithm is implemented (MUST be overriden by subclass)
    stop() to close video feed and windows and stop calling run()

  :ivar _pipeline: a list of frames showing the different steps of the pipeline. It should only ever store the pipeline
                for a single iteration at a given instant (i.e. its length should never exceed the number of steps in
                the pipeline) - not guaranteed to be filled
  :ivar _show_pipeline: a flag indicating whether or not each step in the pipeline should be shown
  :ivar _debug: a flag indicating whether or not the use is debugging the pipeline. In debug, the pipeline is shown and
                debug statements are enabled
  :ivar _capture: the OpenCV capture object (CvCapture) that the lane detection algorithm should run on
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
  """

  SOURCE_SETTINGS_FILE = './source_settings.ini'

  # set up config file reader
  __config = ConfigParser(allow_no_value=True)
  __config.read(path.join(path.dirname(__file__), r'./Pipeline.config'))

  # set up static variables from config file
  SCREEN_WIDTH = int(__config['window']['width'])
  SCREEN_HEIGHT = int(__config['window']['height'])

  MINIMUM_FINAL_IMAGE_RATIO = float(__config['display']['minimum_final_image_ratio'])

  FONT_FACE = vars(cv2)[__config['font']['font_face']]
  FONT_COLOR = tuple(map(int, re.sub('\(|\)| ', '', __config['font']['color']).split(',')))
  FONT_THICKNESS = int(__config['font']['thickness'])
  FONT_SCALE = float(__config['font']['scale'])
  FONT_EDGE_OFFSET = int(__config['font']['edge_offset'])

  NUM_IMAGE_CHANNELS = 3

  def __init__(self, source, should_start, show_pipeline, debug, image_mask_enabled):
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
    self.__stop = False
    self.__source = source

    self.__show_pipeline_steps = Pipeline.__config['display']['show_pipeline_steps'].lower() in ['true', 'yes', '1']
    self.__cached_pipeline = []

    self.__paused = False
    self.__while_paused = None

    class_name = str(self.__class__)
    self._name = class_name[class_name.rindex('.') + 1:-2]
    self._screen = numpy.zeros((Pipeline.SCREEN_HEIGHT, Pipeline.SCREEN_WIDTH, Pipeline.NUM_IMAGE_CHANNELS), numpy.uint8)
    self._pipeline = []

    self._image_mask_enabled = image_mask_enabled
    self._region_of_interest_mask = []

    self._capture = None
    self._show_pipeline = show_pipeline or debug
    self._debug = debug

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
      raise RuntimeError('Cannot open {input} as a capture is already open'.format(input=input))

  def run(self):
    """

    :return: void
    """

    # check if function was called by Process superclass and raise an error if it was not
    if inspect.stack()[1].function != '_bootstrap':
      raise RuntimeError('run method of Pipeline.py can only be invoked by Process superclass')

    # open input
    self.__open_source(self.__source)
    # get fps of video
    fps = self._capture.get(cv2.CAP_PROP_FPS)
    # initialize a flag used to indicate if init_pipeline should be run
    first_frame = True
    # loop run() while the capture is open and we we have not stopped running
    while not self.__stop and self._capture.isOpened():
      # if the pipeline is not paused, read a frame from the capture and call the pipeline
      if not self.__paused:
        # store start time of loop
        start_time = time.time()
        # read a frame of the capture
        return_value, frame = self._capture.read()
        # check that the next frame was read successfully
        # i.e. that we have not hit the end of the video or encountered an error
        if return_value:
          # if it is the first frame of the pipeline, run init_pipeline, then set flag to false
          if first_frame:
            self._init_pipeline(frame)
            first_frame = False
          # run the pipeline
          self._run(frame)
        else:
          # stop the pipeline if we hit the end of the video or encountered an error
          self.stop()
        # only sleep if stop was not called (i.e. we will read the next frame)
        if not self.__stop:
          # 1 second / fps = time to sleep for each frame subtract elapsed time
          time_to_sleep = max(1 / fps - (time.time() - start_time), 0)
          time.sleep(time_to_sleep)
      # if the pipeline is paused and the while_paused handler is defined, call it
      elif self.__while_paused is not None:
        self.__while_paused()

      # get the keypress
      keypress = cv2.waitKey(1) & 0xFF
      # if a key was pressed, call the handler with the pressed key
      if keypress:
        self.__handle_keypress(keypress, frame)

  def __handle_keypress(self, keypress, frame):
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
      self.__show_pipeline_steps = not self.__show_pipeline_steps
    # s - take a screenshot of the pipeline (saves each knot in the pipeline)
    elif keypress == ord('s'):
      self.__take_screenshot()
    # m - allow user to edit mask of source image (if image mask is enabled)
    elif keypress == ord('m') and self._image_mask_enabled:
      self.__modify_region_of_interest(frame)
    # other non-default case (let the subclass handle these cases if desired)
    else:
      self._handle_keypress(keypress, frame)

  @classmethod
  def _handle_keypress(self, keypress, frame):
    """
    @Override - subclass CAN override this function (it is optional)
    Where subclass can add custom keypress events. Cannot override keypress events in Pipeline.py. This inhibits the use
    of the 'q', 'p', and 's' keys and possibly the 'm' key, depending on the state of _image_mask_enabled.

    :param keypress: the code of the keypress (will never correspond to any of the keys used for default keypress events)
    :param frame: the current frame of the pipeline when the keypress occurred
    :return: void
    """

    pass

  def __get_roi_line_index_from_source_settings(self, source_settings):
    """
    Get the line number of the region of interest corresponding to the current source in the given source settings

    :param source_settings: the contents of the source_settings file to search (a .ini file were each section is the
                            source name with an entry entitled roi which is a list of coordinates)
    :return: void
    """

    # iterate through all lines until we find the entry corresponding to the current source to get the next line index
    SOURCE_HEADER = '[{source}]'.format(source=self.__source)
    roi_line_index = 1
    for line in source_settings:
      if line.startswith(SOURCE_HEADER):
        break
      roi_line_index += 1
    # return the region of interest line index if it exists or -1 if it does not
    return roi_line_index if roi_line_index <= len(source_settings) else -1

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
      # open file containing defined image masks and read lines
      with open(Pipeline.SOURCE_SETTINGS_FILE, 'r') as file:
        source_settings = file.readlines()

      # get the line index of the region of interest of the current source
      roi_line_index = self.__get_roi_line_index_from_source_settings(source_settings)
      if roi_line_index != -1:
        # parse the string of coordinates
        string_cords = re.findall('\(\d{1,4}, \d{1,4}\)', source_settings[roi_line_index][5:-2])
        self._region_of_interest_mask = list(map(lambda cord: tuple(map(int, re.sub('\(|\)| ', '', cord).split(','))), string_cords))
      # otherwise get the user to manually defined a mask
      else:
        self.__modify_region_of_interest(first_frame)


  def __modify_region_of_interest(self, frame):
    """
    Gets user to manually modify the region of interest mask, which is stored in a file. User can either keep or discard
    their new changes.

    :param frame: the frame of the pipeline to display that assists the user pick a suitable region of interest
    :raises RuntimeError is raised if this method is called on a pipeline where the image mask is disabled
    :return: void
    """

    # raise an Error if the method is called on a pipeline whose image mask is disabled
    if not self._image_mask_enabled:
      raise RuntimeError('Cannot modify the image mask on a pipeline that has the image mask disabled')

    # create a list to store the new image mask
    new_region_of_interest = []

    def handle_region_of_interest_click(event, x, y, flags, param):
      """
      Handles clicks on the frame to modify the new image mask.

      :param event: the event type of the click
      :param x: the x coordinate of the click event
      :param y: the y coordinate of the click event
      :param flags: unused - see OpenCV documentation (https://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html#setmousecallback)
      :param param: unused - see OpenCV documentation (https://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html#setmousecallback)
      :return: void
      """

      # only handle left clicks
      if event == cv2.EVENT_LBUTTONDOWN:
        # give this method the ability to modify the nonlocal new image mask
        nonlocal new_region_of_interest
        # add the coordinate to the new image mask
        new_region_of_interest.append((x, y))
        # sort the image mask so that the click adds to the existing polygon nicely
        # TODO: Make this better so that points added in between current points do not mess up the shape of the polygon
        new_region_of_interest = sorted(new_region_of_interest, key=lambda cord: cord[0])

    # define keys to keep and discard changes
    CONFIRM_KEY = 'y'
    DENY_KEY = 'n'

    # add instructions to the screen on selecting image mask
    text = "Select Region of Interest - '{confrim}' to confirm changes and '{deny}' to disregard changes".format(confrim=CONFIRM_KEY, deny=DENY_KEY)
    text_bounding_box, text_basline = cv2.getTextSize(text, Pipeline.FONT_FACE, Pipeline.FONT_SCALE, Pipeline.FONT_THICKNESS)
    text_width, text_height = text_bounding_box
    position = (5 + Pipeline.FONT_EDGE_OFFSET, 5 + text_height + Pipeline.FONT_EDGE_OFFSET)
    cv2.putText(frame, text, position, Pipeline.FONT_FACE, Pipeline.FONT_SCALE, Pipeline.FONT_COLOR, Pipeline.FONT_THICKNESS)

    # show the window and add click listener to modify the image mask
    window_name = '{name} - Select Region of Interest'.format(name=self._name)
    cv2.imshow(window_name, frame)
    cv2.setMouseCallback(window_name, handle_region_of_interest_click)

    while True:
      # define an array of polygons (cv2.fillPoly takes an array of polygons)
      # we are only drawing a single polygon so polygons will be a list with a single element (which is the image mask)
      polygons = numpy.empty(shape=(1, len(new_region_of_interest), 2), dtype=numpy.int32)

      # check if there is a coordinate, if there is draw it
      if len(new_region_of_interest):
        # add the image mask to the list of polygons
        polygons[0] = numpy.array(new_region_of_interest)
        # define a base image to draw the polygon on (simply a black screen
        poly_base = numpy.zeros_like(frame)
        # draw the polygons onto the base image
        cv2.fillPoly(poly_base, polygons, (0, 255, 255))
        # show the combination of the frame and polygon image
        cv2.imshow(window_name, cv2.addWeighted(frame, 1, poly_base, 0.5, 0))

      # get a keypress (to see if changes were accepted or rejected)
      keypress = cv2.waitKey(1) & 0xFF
      # if changes were accepted
      if keypress == ord(CONFIRM_KEY):
        # update the region of interest in the pipeline
        self._region_of_interest_mask = new_region_of_interest

        # check if a source settings file exists and make it if necessary
        if not path.exists(Pipeline.SOURCE_SETTINGS_FILE):
          # create the file using with so that resources are cleaned up after creation
          with open(Pipeline.SOURCE_SETTINGS_FILE, 'x'):
            pass

        # read the lines of the source settings file
        with open(Pipeline.SOURCE_SETTINGS_FILE, 'r') as file:
          # read the file using with so that resources are cleaned up after
          source_settings = file.readlines()

        # format the region of interest key
        roi = 'roi=[{list}]\n'.format(list=', '.join(str(cord) for cord in self._region_of_interest_mask))

        # get the line index of the region of interest of the current source
        roi_line_index = self.__get_roi_line_index_from_source_settings(source_settings)
        # if the region of interest mask is defined for the current source, updated it
        if roi_line_index != -1:
          # replace the specific line with the new region of interest
          source_settings[roi_line_index] = roi
        # otherwise add an entry for the current source to the file with the image mask
        else:
          # format the source header
          SOURCE_HEADER = '[{source}]'.format(source=self.__source)
          # add entry for current source to file content
          source_settings.append(SOURCE_HEADER+'\n')
          # add region of interest (image mask) to file content
          source_settings.append(roi)

        # write the modified content to the source settings file
        with open(Pipeline.SOURCE_SETTINGS_FILE, 'w') as file:
          # write to the file using with so that resources are cleaned up after
          file.writelines(source_settings)
        break
      elif keypress == ord(DENY_KEY):
        break

    # remove the mouse listener and destroy the window used for selecting the image mask
    cv2.setMouseCallback(window_name, lambda *args: None)
    cv2.destroyWindow(window_name)

  def _region_of_interest(self, image):
    """
    Applies the region of interest mask to a given image and adds the result to the pipeline. Since the pipeline class
    handles image masking for subclasses (if it is enabled), this method acts as the interface for subclasses to
    automatically apply the image mask to a given step in the pipeline. It is the subclasses responsibility to call this
    method at the appropriate point in the lane detection pipeline.

    :param image: the image to apply the region of interest mask to
    :raises RuntimeError is raised if this method is called on a pipeline where the image mask is disabled
    :return: masked: the image provided with the region of interest mask applied
    """

    # raise an Error if the method is called on a pipeline whose image mask is disabled
    if not self._image_mask_enabled:
      raise RuntimeError('Cannot get the region of interest on a pipeline that has the image mask disabled')

    # define the region of interest as an array of arrays (since cv2.fillPoly takes an array of polygons)
    # we are essentially passing a list with a single entry where that entry is the region of interest mask
    roi = numpy.array([self._region_of_interest_mask])
    # mask is the base image to add the region of interest mask to
    mask = numpy.zeros_like(image)
    # add the region of interst mask to the base image (all black)
    cv2.fillPoly(mask, roi, 255)
    # mask the provided image based on the region of interest
    masked = cv2.bitwise_and(image, mask)
    # add the masked image to the pipeline
    self._add_knot('Region Of Interest Mask', masked)
    # return the masked image
    return masked

  def is_paused(self):
    """
    Gets whether or not the pipeline is paused

    :return: boolean
    """

    return self.__paused

  def _pause(self, while_paused=None):
    """
    Pauses the pipeline and will call the while_paused handler until the pipeline is unpaused

    :raises: RuntimeError is raised if this method is called when the pipeline is already paused
    :return: void
    """

    if self.__paused:
      raise RuntimeError('Cannot pause a pipeline that is already paused')
    self.__paused = True
    self.__while_paused = while_paused

  def _unpause(self):
    """
    Unpauses the pipeline

    :raises: RuntimeError is raised if this method is called when the pipeline is not paused
    :return: void
    """

    if not self.__paused:
      raise RuntimeError('Cannot unpause a pipeline that is not currently paused')
    self.__paused = False
    self.__while_paused = None

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


  def __take_screenshot(self, extension='jpg'):
    """
    Takes a screenshot of the pipeline (saves each knot in the pipeline)

    :param extension (optional) (default=jpg): the file extension that images should be saved as
    :return: void
    """
    # get the directory belonging to this pipeline instance
    pipeline_directory = '{current_directory}/{pipeline_name}_Pipeline'.format(current_directory=path.dirname(__file__), pipeline_name=self._name)
    # make the respective directory if necessary
    if not path.exists(pipeline_directory):
      mkdir(pipeline_directory)
    # make a directory with the current timestamp to contain the screenshots
    current_time_string = datetime.now().strftime('%Y-%m-%d %H%M.%S')
    screenshot_directory = path.join(pipeline_directory, current_time_string)
    mkdir(screenshot_directory)

    # declare a function that calls cv2.imwrite so that it can be threaded
    def write_file_func(file_name, image):
      """
      Wraps cv2.imwrite so that it can be threaded. Also handles some formatting of the image name.

      :param file_name: the name that the image should be saved as
      :param image: the image to be saved
      :return: void
      """

      cv2.imwrite(path.join(screenshot_directory, file_name), image)

    # iterate through each step of the pipeline and save the corresponding image (which is done in a new thread)
    i = 1
    for name, image in self.__cached_pipeline:
      # format the file name
      file_name = '{index}  -  {name}'.format(index=i, name=name) + '.' + extension
      # create a thread and start it
      # intentionally do not join this thread as we want the image writing to happen in the background
      Thread(target=write_file_func, args=(file_name, image)).start()

      i += 1

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
    self.__cached_pipeline = self._pipeline
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
      title = '{index}  -  {name}'.format(index=index, name=name)
      title_bounding_box, title_basline = cv2.getTextSize(title, Pipeline.FONT_FACE, Pipeline.FONT_SCALE, Pipeline.FONT_THICKNESS)
      text_width, text_height = title_bounding_box
      position = (start_x + Pipeline.FONT_EDGE_OFFSET, start_y + text_height + Pipeline.FONT_EDGE_OFFSET)
      cv2.putText(self._screen, title, position, Pipeline.FONT_FACE, Pipeline.FONT_SCALE, Pipeline.FONT_COLOR, Pipeline.FONT_THICKNESS)

    # split the pipeline into the final and intermediate steps
    pipeline_steps = self._pipeline[:-1]
    final_step = self._pipeline[-1]
    num_pipeline_steps = len(pipeline_steps)

    # display the steps of the pipeline only if that option is selected
    if self.__show_pipeline_steps and num_pipeline_steps > 0:
      # initialize the aspect ratio (gets set later when the pipeline is checked for consistent aspect ratios)
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

      # the actual ratio of the final image (will be grater than or equal to Pipeline.MINIMUM_FINAL_IMAGE_RATIO)
      RESULT_IMAGE_RATIO = Pipeline.MINIMUM_FINAL_IMAGE_RATIO
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
        # if this occurs, raise an error and instruct the user to fix the rounding error and update the value in Pipeline.config
        if prev == RESULT_IMAGE_RATIO:
          raise FloatingPointError('Failed trying to find best ratio for result image. This was caused by a floating point decimal error on repeating digits. Update the pipeline.config file and try again. The recomended ratio is {new_ratio} (simply fix the repeating decimals)'.format(new_ratio=RESULT_IMAGE_RATIO))

      # calculate the dimensions of a pipeline step
      container_width = int(round(Pipeline.SCREEN_WIDTH * (1 - RESULT_IMAGE_RATIO)))
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
      output_width = int(round(Pipeline.SCREEN_WIDTH * RESULT_IMAGE_RATIO))
      output_height = int(round(Pipeline.SCREEN_HEIGHT * RESULT_IMAGE_RATIO))
      add_knot_to_screen(len(self._pipeline), knot=final_step, new_dimension=(output_width, output_height), position=(Pipeline.SCREEN_HEIGHT-output_height, Pipeline.SCREEN_WIDTH-output_width))

      cv2.imshow(self._name, self._screen)
    else:
      name, image = final_step
      cv2.imshow(self._name, image)

    # reset the pipeline now that it has been displayed
    self._clear_pipeline()

  def _add_mouse_callback_to_pipeline_window(self, callback):
    """
    Adds the specified mouse callback to the pipeline window

    :param callback: the callback to be added to the pipeline window
    :return: void
    """

    cv2.setMouseCallback(self._name, callback)

  def _remove_mouse_callback_from_pipeline_window(self):
    """
    Removes the mouse callback from the pipeline window

    :return: void
    """

    cv2.setMouseCallback(self._name, lambda *args: None)

  @classmethod
  @abstractmethod
  def _run(self, frame):
    """
    @Override - subclass MUST override this function
    Where the lane detection algorithm is written, it is called on each frame of _capture.

    :param frame: the frame of the capture that the pipeline should be run on
    :return: void
    """

    pass

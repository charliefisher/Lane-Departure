from types import SimpleNamespace

import math
import cv2
import numpy

import settings
import Pipeline


class RegionOfInterest:

  # define keys to keep and discard changes in editor
  EDITOR_CONFIRM_KEY = 'y'
  EDITOR_DENY_KEY = 'n'

  def __init__(self, pipeline: 'Pipeline.Pipeline', when_missing_open_editor: bool = True) -> None:
    self._pipeline = pipeline
    self._when_missing_open_editor = when_missing_open_editor
    self._roi = None

  def get(self) -> list[tuple[int, int]]:
    if self._roi is None:
      raise Exception('Roi does not exist. Call {cls}::load or {cls}::editor'.format(cls=self.__class__.__name__))
    else:
      return self._roi

  def load(self) -> bool:
    rois = settings.load(settings.SettingsCategories.INPUT, settings.InputSettings.ROI, must_exist=False)
    if rois is None or rois.get(self._pipeline.source, None) is None:
      if self._when_missing_open_editor:
        return self.editor(self._pipeline.frame)
      else:
        return False  # failed to load and could not open editor (ROI does not exist)
    else:
      self._roi = rois[self._pipeline.source]
      return True

  def save(self, roi: list[tuple[int, int]]) -> None:
    new_settings = SimpleNamespace()
    new_settings[self._pipeline.source]['roi'] = roi
    settings.save(settings.SettingsCategories.INPUT, settings.InputSettings.ROI, new_settings)

  def editor(self, frame: numpy.array) -> bool:
      """
      Gets user to manually modify the region of interest mask, which is stored in a file. User can either keep or discard
      their new changes.

      :param frame: the frame of the pipeline to display that assists the user pick a suitable region of interest
      :raises RuntimeError is raised if this method is called on a pipeline where the image mask is disabled
      :return: void
      """

      # raise an Error if the method is called on a pipeline whose image mask is disabled
      if not self._pipeline.image_mask_enabled:
        raise RuntimeError('Cannot modify the image mask on a pipeline that has the image mask disabled')

      # create a list to store the new image mask
      new_region_of_interest = []

      def handle_region_of_interest_click(event, x: int, y: int, flags, param):
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
          # add the coordinate to the new image mask
          nonlocal new_region_of_interest
          new_region_of_interest.append((x, y))

      # add instructions to the screen on selecting image mask
      text = "Select Region of Interest - '{confirm}' to confirm changes and '{deny}' to disregard changes"\
             .format(confirm=RegionOfInterest.EDITOR_CONFIRM_KEY, deny=RegionOfInterest.EDITOR_DENY_KEY)
      text_bounding_box, text_baseline = cv2.getTextSize(text, Pipeline.settings.font.face,
                                                         Pipeline.settings.font.scale,
                                                         Pipeline.settings.font.thickness)
      text_width, text_height = text_bounding_box
      position = (5 + Pipeline.settings.font.edge_offset,
                  5 + text_height + Pipeline.settings.font.edge_offset)
      cv2.putText(frame, text, position, Pipeline.settings.font.face, Pipeline.settings.font.scale,
                  Pipeline.settings.font.color, Pipeline.settings.font.thickness)

      # show the window and add click listener to modify the image mask
      window_name = '{name} - Select Region of Interest'.format(name=self._pipeline.name)
      cv2.imshow(window_name, frame)
      cv2.setMouseCallback(window_name, handle_region_of_interest_click)

      while True:
        # define an array of polygons (cv2.fillPoly takes an array of polygons)
        # we are only drawing a single polygon so polygons will be a list with a single element
        # (which is the image mask)
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
        if keypress == ord(RegionOfInterest.EDITOR_CONFIRM_KEY):
          # update the region of interest in the pipeline
          settings.save(settings.SettingsCategories.INPUT, settings.InputSettings.ROI,
                        {self._pipeline.source: new_region_of_interest})
          updated = True
          break
        elif keypress == ord(RegionOfInterest.EDITOR_DENY_KEY):
          updated = False
          break

      # remove the mouse listener and destroy the window used for selecting the image mask
      cv2.setMouseCallback(window_name, lambda *args: None)
      cv2.destroyWindow(window_name)
      return updated


class Visualizer:
  '''

  :friend_of Pipeline.Pipeline
  '''

  def __init__(self, pipeline: 'Pipeline.Pipeline') -> None:
    self._pipeline = pipeline
    self._has_init = False

    # initialize variables concerned with the size of pipeline step bins
    # (will get set later when calculating self.result_image_ratio)
    self._horizontal_bins_dimension = 0
    self._vertical_bins_dimension = 0

    self._result_image_ratio = 1

    # initialize the aspect ratio (gets set later when the pipeline is checked for consistent aspect ratios)
    self._aspect_ratio = None

    # initialize the pipeline steps
    self._knots = []
    self._num_steps = 0

    self._container_width = 0
    self._step_width = 0
    self._step_height = 0

  def get(self):
    """
    Displays the pipeline to the user. Depending on the state of _show_pipeline_steps, the steps of the pipeline may
    or may not be visible.

    :return: void
    """

    # copy the pipeline knots (as they may be modified before being added to screen)
    self._knots = self._pipeline.friend_access(self, '__current_knots')
    self._num_steps = len(self._knots) - 1

    # display the steps of the pipeline only if that option is selected
    if self._pipeline.show_pipeline_steps and self._num_steps > 0:
      self._check_aspect_ratios_and_fix_num_channels()
      if not self._has_init:
        self._determine_knot_image_size()

        # calculate the dimensions of a pipeline step
        self._container_width = int(round(Pipeline.settings.window.width * (1 - self._result_image_ratio)))
        self._step_width = self._container_width // self._horizontal_bins_dimension
        self._step_height = int(round(self._step_width * self._aspect_ratio))

      # iterate through all but the final step and display those knots in the pipeline
      for i, step in enumerate(self._knots[:-1]):
        # add the knot to the screen at the correct position
        start_y = self._step_height * (i // self._horizontal_bins_dimension)
        start_x = self._step_width * (i % self._horizontal_bins_dimension)
        self._add_knot_to_image(i + 1, knot=step, new_dimension=(self._step_width, self._step_height),
                                position=(start_y, start_x))

      # add the final step to the screen in the bottom left quarter
      output_width = int(round(Pipeline.settings.window.width * self._result_image_ratio))
      output_height = int(round(Pipeline.settings.window.height * self._result_image_ratio))
      self._add_knot_to_image(len(self._knots), knot=self._knots[-1],
                              new_dimension=(output_width, output_height),
                              position=(Pipeline.settings.window.height - output_height,
                                        Pipeline.settings.window.width - output_width))

      cv2.imshow(self._pipeline.name, self._pipeline.friend_access(self, '_screen'))
    else:  # only one step or show pipeline steps is disabled
      name, image = self._knots[-1]  # final step
      cv2.imshow(self._pipeline.name, image)

  def _check_aspect_ratios_and_fix_num_channels(self):
    # check that all steps of the pipeline have the same aspect ratio (if not raise and error)
    # simultaneously, check if any images are single channel and convert them to the correct number of channels
    for i, knot in enumerate(self._pipeline.friend_access(self, '__current_knots')):
      name, image = knot
      # get the dimensions of the image
      # note that if the image is single channel, then num_channels will be undefined -> set it to default value after
      height, width, *num_channels = image.shape
      num_channels = num_channels[0] if num_channels else 1

      # check for aspect ratio consistency throughout the pipeline
      cur_aspect_ratio = height / width
      if self._aspect_ratio is None:
        self._aspect_ratio = cur_aspect_ratio
      elif cur_aspect_ratio != self._aspect_ratio:
        raise RuntimeError('Aspect Ratio of images is not consistent throughout pipeline')

      # if the image is single channel (grayscale), convert it to 3 channels (still grayscale)
      # this allows the images to be merged into one
      if num_channels == 1:
        temp_image = numpy.empty((height, width, Pipeline.NUM_IMAGE_CHANNELS))
        for channel in range(Pipeline.NUM_IMAGE_CHANNELS):
          temp_image[:, :, channel] = image

        self._knots[i] = (name, temp_image)

  def _calculate_dimensions_given_ratio(self, ratio):
    """
    Calculates pipeline step bin dimensions given a ratio for the final step of the pipeline

    :param ratio: the ratio of the final step of the pipeline to the rest of the screen
    :return: void
    """

    def next_square(num: int) -> int:
      """
      Calculate the next lowest square greater than num

      :param num: the number that the square must be greater than
      :return: int
      """
      return int(round(math.pow(math.ceil(math.sqrt(abs(num))), 2)))

    # calculate the size of the step bins
    # the final result is displayed in the bottom corner, thus calculate the number of binds the must fit in the top
    # left corner
    num_bins_top_left = next_square(math.ceil(self._num_steps * (1 - ratio)))
    # using the number of bins in the top left corner, get the number of bins along the horizontal and vertical
    self._horizontal_bins_dimension = int(round(math.sqrt(num_bins_top_left)))
    self._vertical_bins_dimension = math.pow(1 - ratio, -1) * self._horizontal_bins_dimension

  def _determine_knot_image_size(self):
    # the actual ratio of the final image (will be grater than or equal to
    # Pipeline.settings.display.minimum_final_image_ratio)
    self._result_image_ratio = Pipeline.settings.display.minimum_final_image_ratio

    # mimic a do-while loop
    while True:
      # calculate the bin dimensions for the current ratio
      self._calculate_dimensions_given_ratio(self._result_image_ratio)
      # if the number of vertical bins is an integer (divides evenly into the screen), then break the loop
      # (the while condition of the do-while loop)
      if self._vertical_bins_dimension.is_integer():
        break
      # store the previously calculated ratio
      prev = self._result_image_ratio
      # calculate the new ratio to use
      self._result_image_ratio = 1 - self._horizontal_bins_dimension / math.ceil(self._vertical_bins_dimension)
      # due to floating point precision errors, sometimes repeating decimals get rounded in an undesirable manner
      # essentially, the program has successfully found the desired ratio, but rounds it causing the program to fail
      # if this occurs, raise an error and instruct the user to fix the rounding error and update the value in
      # Pipeline settings
      if prev == self._result_image_ratio:
        raise FloatingPointError('Failed trying to find best ratio for result image. This was caused by a floating ' +
                                 'point decimal error on repeating digits. Update the pipeline.config file and try ' +
                                 'again. The recommended ratio is {new_ratio} (simply fix the repeating decimals)'
                                 .format(new_ratio=self._result_image_ratio))

  def _add_knot_to_image(self, index, knot, new_dimension, position):
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
    self._pipeline.friend_access(self, '_screen')[start_y:(start_y + height), start_x:(start_x + width)] = resized_image

    # add the title of the knot to the image
    title = '{index} - {name}'.format(index=index, name=name)
    title_bounding_box, title_basline = cv2.getTextSize(title, Pipeline.settings.font.face,
                                                        Pipeline.settings.font.scale,
                                                        Pipeline.settings.font.thickness)
    text_width, text_height = title_bounding_box
    position = (start_x + Pipeline.settings.font.edge_offset,
                start_y + text_height + Pipeline.settings.font.edge_offset)
    cv2.putText(self._pipeline.friend_access(self, '_screen'), title, position, Pipeline.settings.font.face,
                Pipeline.settings.font.scale, Pipeline.settings.font.color, Pipeline.settings.font.thickness)

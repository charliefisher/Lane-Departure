from typing import Callable, Optional

import cv2
import numpy as np


class Visualizer:
  def __init__(self,
               window_name: str,
               get_frame: Callable[[], np.array],
               visualize_function: Callable[[], None],
               init_function: Optional[Callable[[], None]] = None,
               click_listener: Optional[Callable[[int, int, int, dict, dict], None]] = None,
               handle_keypress: Optional[Callable[[int], None]] = None,
               deinit_function: Optional[Callable[[], None]] = None) -> None:
    self.window_name = window_name
    self._get_frame = get_frame
    self._init_function = init_function
    self._click_listener = click_listener
    self._visualize_function = visualize_function
    self._handle_keypress = handle_keypress
    self._deinit_function = deinit_function

  def run(self, window_name):
    # do whatever setup we need for the window
    if self._init_function is not None:
      self._init_function()

    # add click listener
    if self._click_listener is not None:
      cv2.setMouseCallback(window_name, self._click_listener)

    while True:
      frame = self._get_frame()
      cv2.imshow(window_name, frame)

      self._visualize_function()

      if self._handle_keypress is not None:
        keypress = cv2.waitKey(1) & 0xFF
        self._handle_keypress(keypress)

    # remove the mouse listener and destroy the window used for selecting the image mask
    if self._deinit_function is not None:
      self._deinit_function()
    if self._click_listener is not None:
      cv2.setMouseCallback(window_name, lambda *args: None)
    cv2.destroyWindow(window_name)





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

      # define keys to keep and discard changes
      CONFIRM_KEY = 'y'
      DENY_KEY = 'n'

      # add instructions to the screen on selecting image mask
      text = "Select Region of Interest - '{confrim}' to confirm changes and '{deny}' to disregard changes".format(
        confrim=CONFIRM_KEY, deny=DENY_KEY)
      text_bounding_box, text_basline = cv2.getTextSize(text, Pipeline.FONT_FACE, Pipeline.FONT_SCALE,
                                                        Pipeline.FONT_THICKNESS)
      text_width, text_height = text_bounding_box
      position = (5 + Pipeline.FONT_EDGE_OFFSET, 5 + text_height + Pipeline.FONT_EDGE_OFFSET)
      cv2.putText(frame, text, position, Pipeline.FONT_FACE, Pipeline.FONT_SCALE, Pipeline.FONT_COLOR,
                  Pipeline.FONT_THICKNESS)

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
            source_settings.append(SOURCE_HEADER + '\n')
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
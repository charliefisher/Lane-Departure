from types import SimpleNamespace
from typing import Optional

import cv2
import numpy

import settings
import Pipeline.Pipeline as Pipeline


class RegionOfInterest:
  def __init__(self, pipeline: Pipeline, when_missing_open_editor: bool = True) -> None:
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

  def save(self, roi):
    new_settings = SimpleNamespace()
    new_settings[self._pipeline.source]['roi'] = roi
    settings.save(settings.SettingsCategories.INPUT, settings.InputSettings.ROI, new_settings)

  def editor(self, frame) -> bool:
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
          # add the coordinate to the new image mask
          nonlocal new_region_of_interest
          new_region_of_interest.append((x, y))

      # define keys to keep and discard changes
      CONFIRM_KEY = 'y'
      DENY_KEY = 'n'

      # add instructions to the screen on selecting image mask
      text = "Select Region of Interest - '{confirm}' to confirm changes and '{deny}' to disregard changes"\
             .format(confirm=CONFIRM_KEY, deny=DENY_KEY)
      text_bounding_box, text_baseline = cv2.getTextSize(text, Pipeline.Pipeline.settings.font.face,
                                                         Pipeline.Pipeline.settings.font.scale,
                                                         Pipeline.Pipeline.settings.font.thickness)
      text_width, text_height = text_bounding_box
      position = (5 + Pipeline.Pipeline.settings.font.edge_offset,
                  5 + text_height + Pipeline.Pipeline.settings.font.edge_offset)
      cv2.putText(frame, text, position, Pipeline.Pipeline.settings.font.face, Pipeline.Pipeline.settings.font.scale,
                  Pipeline.Pipeline.settings.font.color, Pipeline.Pipeline.settings.font.thickness)

      # show the window and add click listener to modify the image mask
      window_name = '{name} - Select Region of Interest'.format(name=self._pipeline.name)
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
          settings.save(settings.SettingsCategories.INPUT, settings.InputSettings.ROI,
                        {self._pipeline.source: new_region_of_interest})
          updated = True
          break
        elif keypress == ord(DENY_KEY):
          updated = False
          break

      # remove the mouse listener and destroy the window used for selecting the image mask
      cv2.setMouseCallback(window_name, lambda *args: None)
      cv2.destroyWindow(window_name)
      return updated

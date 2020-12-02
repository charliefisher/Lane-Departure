import cv2
import numpy


def region_of_interest(pipeline, image):
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
  if not pipeline.image_mask_enabled:
    raise RuntimeError('Cannot get the region of interest on a pipeline that has the image mask disabled')

  # define the region of interest as an array of arrays (since cv2.fillPoly takes an array of polygons)
  # we are essentially passing a list with a single entry where that entry is the region of interest mask
  roi = numpy.array([pipeline.region_of_interest])
  # mask is the base image to add the region of interest mask to
  mask = numpy.zeros_like(image)
  # add the region of interst mask to the base image (all black)
  cv2.fillPoly(mask, roi, 255)
  # mask the provided image based on the region of interest
  masked = cv2.bitwise_and(image, mask)
  # return the masked image
  return masked

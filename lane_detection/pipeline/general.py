"""
A collection of general functions used in image processing pipelines.

Any logic that is not pipeline specific should reside in this file.
"""


import cv2
import numpy


def region_of_interest(pipeline, image) -> numpy.array:
  """
  Applies the region of interest mask to a given image

  :param pipeline: the pipeline the region of interest mask is applied to
  :param image: the image to apply the region of interest mask to
  :raises RuntimeError is raised if this method is called on a pipeline where the image mask is disabled
  :return: numpy.array: the image provided with the region of interest mask applied
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
  n_channels = mask.shape[2] if len(mask.shape) == 3 else 1
  cv2.fillPoly(mask, roi, (255,) * n_channels)
  # mask the provided image based on the region of interest
  masked = cv2.bitwise_and(image, mask)
  # return the masked image
  return masked

import os
from configparser import ConfigParser
from argparse import ArgumentParser

from lane_detection.HoughTransform import HoughTransform
from lane_detection.HistogramPeakDetection import HistogramPeakDetection


# global variables that dictate how the lane detection lane_detection should be run
camera = None
file = None
show_pipeline = False
debug = False

# set up config file reader
__config = ConfigParser(allow_no_value=True)
__config.read(r'./config.ini')

def __main():
  """
  Configures run options based off of arguments and starts lane_detection

  :return: void
  """

  if camera is not None:   # use camera as input
    source = camera
  elif file is not None:   # use file as input
    source = file
  else:   # use default as input
    defaultInput = __config['default']['input_source']
    if defaultInput == 'file':
      source = __get_abs_path(__config['default']['file_source'])
    elif defaultInput == 'camera':
      camera_type = __config['default']['camera_source']
      source = int(__config['cameras'][camera_type])
    else:
      raise Exception('invalid default input type')

  hough = HoughTransform(source, True, True, debug)
  histopeak = HistogramPeakDetection(source, True, True, debug)

def __parse_args():
  """
  Parses the command line arguments and sets necessary global flags accordingly

  If no arguments are given, the defaults are used according to config.ini

  Arguments:
    --show-pipeline | -p    indicates whether each processing step in the pipeline should be displayed
    --debug                 indicates whether debug statements should be printed and shows pipeline

    Mutually Exclusive:     whichever argument is given is the input source used
      --camera | -c         selects which camera should be used for input [webcam (w) or usb (u)]
      --file | -f           path of file to use as input

  :return: void
  """

  # add global keyword so we can modify global variables (dangerous, but make sense in this case)
  global camera, file, show_pipeline, debug

  # create argument parser and then parse them
  parser = ArgumentParser()
  parser.add_argument('--show-pipeline', '-p', action='store_true')
  parser.add_argument('--debug', action='store_true')
  inputType = parser.add_mutually_exclusive_group(required=False)
  inputType.add_argument('--camera', '-c', nargs=1, choices=['webcam', 'w', 'usb', 'u'], type=str)
  inputType.add_argument('--file', '-f', nargs=1, type=str)
  args = parser.parse_args()

  # format arguments and set global variables
  if args.camera is not None:
    if args.camera[0] == 'webcam' or args.camera[0] == 'w':
      camera = 'webcam'
    elif args.camera[0] == 'usb' or args.camera[0] == 'u':
      camera = 'usb'
    else:
      raise Exception('failed to set camera input')
  elif args.file is not None:
    file = __get_abs_path(args.file[0])
  show_pipeline = args.show_pipeline
  debug = args.debug


def __get_abs_path(path):
  """
  Gets the absolute path of a file

  This method is susceptible to raise any errors caused by os.path.abspath(path)

  :param path: The file path to be made absolute (can be either relative or absolute)
  :raises: Exception is raised if the given file path does not exist
  :return: The absolute version of the given path
  :rtype: str
  """

  # check if the file path exists
  if os.path.exists(path):
    # get the absolute file path
    return os.path.abspath(path)
  raise Exception('given file path does not exist')

if __name__ == '__main__':
  """
  Parses arguments and calls the main() function
  
  Invoked when the script is called from the command line via 'python main.py'
  """

  # parse arguments
  __parse_args()
  # call main()
  __main()

import os
import configparser
import argparse


camera = None
file = None
defaultInput = None
showPipeline = False


def main():
    print(camera, file, defaultInput, showPipeline)


def parse_args():
    # add global keyword so we can modify global variables (dangerous, but make sense in this case)
    global camera, file, defaultInput, showPipeline

    # create argument parser and then parse them
    parser = argparse.ArgumentParser()
    parser.add_argument('--show-pipeline', '-p', action='store_true')
    inputType = parser.add_mutually_exclusive_group(required=False)
    inputType.add_argument('--camera', '-c', nargs=1, choices=['webcam', 'w', 'usb', 'u'], type=str)
    inputType.add_argument('--file', '-f', nargs=1, type=str)
    args = parser.parse_args()

    # set up config file reader
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(r'./config.ini')

    # format arguments and set global variables
    if args.camera is not None:
        if args.camera[0] == 'webcam' or args.camera[0] == 'w':
            camera = 'webcam'
        elif args.camera[0] == 'usb' or args.camera[0] == 'u':
            camera = 'usb'
        else:
            raise Exception('failed to set camera input')
    if args.file is not None:
        if os.path.exists(args.file[0]):
            file = os.path.abspath(args.file[0])
        else:
            raise Exception('given file path does not exist')
    defaultInput = config['default']['input_source']
    showPipeline = args.show_pipeline


# calls the main() function
if __name__ == '__main__':
  parse_args()
  main()

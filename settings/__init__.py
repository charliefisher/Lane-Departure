import os
from enum import Enum
from typing import Any, Optional, Union
import functools

from config import Config

from general.config_dict import config_dict, read_only_config_dict, merge

__all__ = ['load', 'save', 'SettingsCategories', 'InputSettings', 'PipelineSettings']


class SettingsCategories(Enum):
  GENERAL = 'general'
  INPUT = 'input'
  PIPELINES = 'pipelines'


class InputSettings(Enum):
  SOURCES = 'sources'
  ROI = 'region_of_interest'


class PipelineSettings(Enum):
  HOUGH_TRANSFORM = 'hough_transform'
  HISTOGRAM_PEAK_DETECTION = 'histogram_peak_detection'

Settings = Union[InputSettings, PipelineSettings]


SETTINGS_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
DEFAULT_SETTING_FILE_EXTENSION = 'cfg'


def load(category: SettingsCategories, settings: Settings, must_exist: bool = True) -> Optional[config_dict]:
  settings_file = _get_file_for_settings(category, settings)
  if not os.path.exists(settings_file):
    if must_exist:
      raise FileExistsError('Settings file {f} does not exist'.format(f=settings_file))
    else:
      return None
  settings = Config(settings_file).as_dict()
  __code_types(settings, __CodeTypesOpts.DECODE)
  return __convert_to_config_dict(settings)


def save(category: SettingsCategories, settings: Settings, new_settings: Union[dict, config_dict]) -> bool:
  current_settings = load(category, settings, must_exist=False)
  if current_settings is not None:
    new_settings = merge(current_settings, new_settings)

  with open(_get_file_for_settings(category, settings), 'w') as f:
    settings_dict = new_settings if isinstance(new_settings, dict) else new_settings.__dict__
    __code_types(settings_dict, __CodeTypesOpts.ENCODE)
    print(__print_formatter(settings_dict), end='', file=f)
    return True
  return False


def _get_file_for_settings(category: SettingsCategories, settings: Settings) -> str:
  category_directory = os.path.join(SETTINGS_DIRECTORY, category.value)
  os.makedirs(category_directory, exist_ok=True)
  return os.path.join(category_directory, '{setting}.{ext}'.format(setting=settings.value,
                                                                   ext=DEFAULT_SETTING_FILE_EXTENSION))


def _decoder(x: Any) -> Any:
  if isinstance(x, str) and (x.startswith('(') and x.endswith(')')):  # tuple
    return eval(x)
  else:
    return x


def _encoder(x: Any) -> Any:
  if isinstance(x, tuple):
    return "'{tuple}'".format(tuple=x)
  else:
    return x


# return type could be config_dict since read_only_config_dict is a subclass but we list as Union here to be explicit
def __convert_to_config_dict(setting: dict) -> Union[config_dict, read_only_config_dict]:
  runtime_editable = setting.pop('runtime_editable', True)
  if runtime_editable:
    return config_dict(setting)
  else:
    return read_only_config_dict(setting)


class __CodeTypesOpts(Enum):
  ENCODE = functools.partial(_encoder)
  DECODE = functools.partial(_decoder)


def __code_types(setting: dict, coder: __CodeTypesOpts) -> None:
  assert isinstance(coder, __CodeTypesOpts)

  for k in setting.keys():
    if isinstance(setting[k], dict):  # recurse for dict case
      __code_types(setting[k], coder)
    elif isinstance(setting[k], list):
      for i in range(len(setting[k])):  # iterate over list elements
        if isinstance(setting[k][i], dict):  # recurse for dict case
          __code_types(setting[k][i], coder)
        else:
          setting[k][i] = coder.value(setting[k][i])   # encode / decode
    else:
      setting[k] = coder.value(setting[k])  # encode / decode


def __print_formatter(obj, indent=2, width=80) -> str:

  def print_formatter_internal(recursion_level=0):
    result = ''
    for k, v in obj.items():
      if isinstance(v, dict):
        value = '{{\n{sub_dict}\n}}'.format(sub_dict=print_formatter_internal(x, recursion_level + 1))
      elif isinstance(v, list):
        value = ','.join(v)
        if len(str(k)) + len(value) + 2 > width:
          value = '[\n'
          for x in v:
            if isinstance(x, dict) or isinstance(x, list):
              value += print_formatter_internal(x, recursion_level + 1)
            else:
              value += ' ' * indent * (recursion_level + 1) + '{value},\n'.format(value=x)
          # do not need a newline preceding the closing brace since each value prints a new line following itself
          value += ']'
      else:
        value = v

      result += ' ' * indent * recursion_level + "'{key}': {value},\n".format(key=k, value=value)

    return result

  return print_formatter_internal(obj, 0).replace('\\', '\\\\')

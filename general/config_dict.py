from dataclasses import dataclass
from copy import deepcopy

from attrdict import AttrDict
from deepmerge import Merger


config_dict = AttrDict


@dataclass(init=False, frozen=True)
class read_only_config_dict(config_dict):
  pass


def merge(base: config_dict, overrides: config_dict, allow_in_place: bool = True) -> config_dict:
  merger = Merger(
    [(dict, ["merge"])],  # strategies to use for each type
    ["override"],  # fallback strategies applied to all other types
    ["override"]  # strategies when the types conflict
  )

  if not allow_in_place:
    base = deepcopy(base)
    overrides = deepcopy(overrides)

  merger.merge(base, overrides)
  return base

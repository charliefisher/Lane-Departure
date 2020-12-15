from typing import Any, Type


def register_friend(friend: Any) -> Any:
  # need an interior decorator so we can accept arguments in the outer decorator
  def decorator(cls):
    assert issubclass(cls, Friendable)  # class must be Friendable to register a friend
    cls.register_friend(friend)  # register the friend
    return cls  # return the class from the decorator
  return decorator


class Friendable:
  # a set of tuples: the first entry is who we are a friend of, and the second entry is the friend
  # this is used to allow friend access up to the inheritance tree (but not higher than our friend)
  __friends = set()

  @classmethod
  def register_friend(cls, friend: Any) -> Any:
    cls.__friends.add((cls, friend))

  def friend_access(self, caller: Type, item: str) -> Any:
    if item == '__friends':  # do not allow access to the list of friends
      raise AttributeError

    # iterate over list of friends until we find the caller
    for friend_of, friend in Friendable.__friends:
      if not isinstance(caller, friend):
        continue  # not called by this friend, skip
      else:  # we accessing from a friend class
        # iterate over classes in the method resolution order up to and including the class the caller is a friend of
        for class_ in self.__class__.mro():
          try:
            # handle attribute name mangling for private attributes
            item_mangled = item
            if item.startswith('__'):
              item_mangled = '_{class_name}{item}'.format(class_name=class_.__qualname__, item=item)

            # try to read the attribute from the class
            return class_.__getattribute__(self, item_mangled)

          # catch failures from __getattribute__ where attribute does not exist in class
          except AttributeError:
            if class_ == friend_of:  # we have hit the highest we can go in the method resolution order
              break  # will cause us to raise AttributeError below
            pass

    # either this method was not called by a friend, or
    # the attribute does not exist in the class or one of its subclasses (up to the type itself)
    raise AttributeError

from typing import Any, Callable, Type


def register_friend(friend: Any) -> Callable[[Any], Any]:
  """
  A decorator used by Friendable classes to register friends

  :param friend: the friend class to register
  :return: Callable[[T], T]
  """

  # need an interior decorator so we can accept arguments in the outer decorator
  def decorator(cls):
    assert issubclass(cls, Friendable)  # class must be Friendable to register a friend
    cls.register_friend(friend)  # register the friend
    return cls  # return the class from the decorator
  return decorator


class Friendable:
  """
  A class that mimics the behaviour of C++ friend classes.

  A class that can have friends. A friend class can access the private instance variables of the class it is friends
  with.

  Use: subclass Friendable to allow friend access to the class' private instance variables. To add friends, use the
  register_friend decorator or call Friendable::register_friend.

  This implementation allows access to the class' subclasses' private instance variables. In theory, a friend should not
  have access to a subclass' private instance variables. However, if the object is type-casted by the friend at
  runtime, this is a safe access. Since we cannot type-cast at runtime in all scenarios in Python (for example, if the
  class is abstract), we assume this form of access is allowed.

  Further to this point, friend classes are un-Pythonic. Allowing some leniency in what the friend class can access does
  not break the language constructs (anymore than having a friend class does).
  """

  # a set of tuples: the first entry is who we are a friend of, and the second entry is the friend
  # this is used to allow friend access down the inheritance tree (but not higher than our friend)
  # NOTE: the inheritance tree is actually traversed upwards, but this is an implementation detail
  __friends = set()

  @classmethod
  def register_friend(cls, friend: Any) -> None:
    """
    Registers a friend class

    :param friend: the friend class to register
    :return: void
    """

    cls.__friends.add((cls, friend))

  def friend_access(self, caller: Type, item: str) -> Any:
    """
    Access a private (or public) instance variable from a friend class

    :param caller: the instance calling this function
    :param item: the attribute to access
    :raises: AttributeError is raised if the attribute does not exist or if this method was not called by a friend
    :return: Any
    """

    if item == '__friends':  # do not allow access to the list of friends
      raise AttributeError(item)

    # iterate over list of friends until we find the caller
    for friend_of, friend in Friendable.__friends:
      if not isinstance(caller, friend):
        continue  # not called by this friend, skip
      else:  # accessing from a friend class
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
    raise AttributeError(item)

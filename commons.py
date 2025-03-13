"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import reduce
from operator import or_
from typing import Any, TypeVar


@dataclass(init=False)
class AbstractDataclass(ABC):
    """
    An abstract base class for creating dataclasses.
    This class provides a basic structure for creating dataclasses, ensuring that
    all subclasses implement their own `__init__` method. It does not generate an
    `__init__` method by default to allow for custom initialization logic in
    subclasses.

    Note that the `init=False` parameter is explicitly set here, which is the
    opposite of the default behavior of the `@dataclass` decorator, where
    `init=True` by default. By setting `init=False`, we prevent the automatic
    generation of an `__init__` method in the subclass, allowing the subclass
    to define its own `__init__` method. The abstract `__init__` method defined here
    must be implemented by all subclasses, either by allowing the `@dataclass` decorator
    to auto-generate it (by not setting `init=False`) or by providing a manual implementation.

    If you want to keep this abstract property in your subclasses, make sure to set
    `init=False` in your subclass definition as well; otherwise, `@dataclass`
    automatically generates an `__init__` method to make it a concrete dataclass.

    Following is the example usage:
    ```
    @dataclass(init=False)
    class ChildAbstractDataclass(AbstractDataclass):
    # Not able to instantiate this dataclass.

    @dataclass(init=False)
    class GrandchildAbstractDataclass(ChildAbstractDataclass):
    # Still not able to instantiate this dataclass.

    @dataclass
    class EmptyConcreteDataclass(AbstractDataclass):
    # A dataclass with no field.
    ```

    """

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """An abstract method that must be implemented by all subclasses."""
        ...


SubclassesType = TypeVar("SubclassesType")


def get_all_subclasses(
    cls: SubclassesType, include_cls_self: bool = True
) -> list[SubclassesType]:
    """
    Retrieves all subclasses of a given class, optionally including the class itself.

    This function uses a helper function to recursively find all unique subclasses
    of the specified class.

    Args:
        cls (SubclassesType): The class for which to find subclasses.
        include_cls_self (bool): Whether to include the class itself in the result. (Default: True)

    Returns:
        list[SubclassesType]: A list of all unique subclasses of the given class.
    """

    def get_all_unique_subclasses(cls: SubclassesType) -> set[SubclassesType]:
        """Gets all unique subclasses of a given class recursively."""
        assert (
            subclasses := getattr(cls, "__subclasses__", lambda: None)()
        ) is not None, f"{cls} does not have __subclasses__."
        return reduce(or_, map(get_all_unique_subclasses, subclasses), {cls})

    return list(get_all_unique_subclasses(cls) - (set() if include_cls_self else {cls}))

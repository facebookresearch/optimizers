"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from functools import reduce
from itertools import islice
from operator import methodcaller, or_
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


_SubclassesType = TypeVar("_SubclassesType")


def get_all_non_abstract_subclasses(cls: _SubclassesType) -> Iterable[_SubclassesType]:
    """
    Retrieves all non-abstract (instantiable) subclasses of a given class.

    This function uses a helper function to recursively find all unique subclasses
    of the specified class, and then filters out any abstract classes.

    Args:
        cls (_SubclassesType): The class for which to find subclasses.

    Returns:
        non_abstract_subclasses (Iterable[_SubclassesType]): An iterable of all unique non-abstract subclasses of the given class.
    """

    def get_all_unique_subclasses(cls: _SubclassesType) -> set[_SubclassesType]:
        """Gets all unique subclasses of a given class recursively."""
        return reduce(
            or_,
            map(get_all_unique_subclasses, methodcaller("__subclasses__")(cls)),
            {cls},
        )

    return filter(
        # Filters out abstract classes by checking if '__abstractmethods__' is an empty set or not present.
        lambda sub_cls: not getattr(sub_cls, "__abstractmethods__", frozenset()),
        get_all_unique_subclasses(cls),
    )


_BatchedInputType = TypeVar("_BatchedInputType")


def batched(
    iterable: Iterable[_BatchedInputType], n: int
) -> Iterable[tuple[_BatchedInputType, ...]]:
    """
    Batches an iterable into chunks of size n.

    Note: This is a forward implementation of itertools.batched which is available in Python 3.12+.
    Remove this function when downstream applications are using Python 3.12 or newer.

    Args:
        iterable (Iterable[_BatchedInputType]): The iterable to be batched.
        n (int): The size of each batch.

    Yields:
        batched_tuple (tuple[_BatchedInputType, ...]): A generator that yields batches of size n.

    Raises:
        ValueError: If n is less than 1.
    """
    if n < 1:
        raise ValueError(f"{n=} must be at least one")

    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch

"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

from functools import reduce
from operator import or_
from typing import TypeVar

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

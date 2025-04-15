"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import re
import unittest

from dataclasses import dataclass

from commons import AbstractDataclass, get_all_subclasses
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


@instantiate_parametrized_tests
class InvalidAbstractDataclassInitTest(unittest.TestCase):
    @dataclass(init=False)
    class DummyOptimizerConfig(AbstractDataclass):
        """Dummy abstract dataclass for testing. Instantiation should fail."""

    @parametrize("abstract_cls", (AbstractDataclass, DummyOptimizerConfig))
    def test_invalid_init(self, abstract_cls: type[AbstractDataclass]) -> None:
        self.assertRaisesRegex(
            TypeError,
            re.escape(f"Can't instantiate abstract class {abstract_cls.__name__} "),
            abstract_cls,
        )


class DummyRootClass:
    """Dummy root class for GetAllSubclassesTest."""


class DummyFirstSubclass(DummyRootClass):
    """First dummy subclass for GetAllSubclassesTest."""


class DummySecondSubclass(DummyFirstSubclass):
    """Second dummy subclass for GetAllSubclassesTest."""


class DummySecondRootClass:
    """Second dummy root class for GetAllSubclassesTest."""


class DummyMixedSubclass(DummySecondRootClass, DummySecondSubclass):
    """Dummy subclass with mixed inheritance for GetAllSubclassesTest."""


class DummyLeafClass(DummyMixedSubclass):
    """Dummy leaf class for GetAllSubclassesTest."""


@instantiate_parametrized_tests
class GetAllSubclassesTest(unittest.TestCase):
    @parametrize("include_cls_self", (True, False))
    def test_class_hierarchy_and_multiple_inheritance(
        self, include_cls_self: bool
    ) -> None:
        subclasses = [
            DummyFirstSubclass,
            DummySecondSubclass,
            DummyMixedSubclass,
            DummyLeafClass,
        ]
        self.assertCountEqual(
            get_all_subclasses(DummyRootClass, include_cls_self=include_cls_self),
            subclasses + [DummyRootClass] * include_cls_self,
        )

    @parametrize("include_cls_self", (True, False))
    def test_second_subclass(self, include_cls_self: bool) -> None:
        subclasses = [DummyMixedSubclass, DummyLeafClass]
        self.assertCountEqual(
            get_all_subclasses(DummySecondSubclass, include_cls_self=include_cls_self),
            subclasses + [DummySecondSubclass] * include_cls_self,
        )

    @parametrize("include_cls_self", (True, False))
    def test_leaf_class(self, include_cls_self: bool) -> None:
        self.assertCountEqual(
            get_all_subclasses(DummyLeafClass, include_cls_self=include_cls_self),
            [DummyLeafClass] * include_cls_self,
        )

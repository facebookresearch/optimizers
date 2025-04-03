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


@dataclass(init=False)
class DummyOptimizerConfig(AbstractDataclass):
    """Dummy abstract dataclass for testing. Instantiation should fail."""


class InvalidAbstractDataclassInitTest(unittest.TestCase):
    def test_invalid_init(self) -> None:
        for abstract_cls in (
            AbstractDataclass,
            DummyOptimizerConfig,
        ):
            with self.subTest(abstract_cls=abstract_cls):
                self.assertRaisesRegex(
                    TypeError,
                    re.escape(
                        f"Can't instantiate abstract class {abstract_cls.__name__} "
                    ),
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


class GetAllSubclassesTest(unittest.TestCase):
    def test_get_all_subclasses(self) -> None:
        """Test with class hierarchy and multiple inheritance."""
        for include_cls_self in (True, False):
            with self.subTest(
                "Test with the root class", include_cls_self=include_cls_self
            ):
                subclasses = {
                    DummyFirstSubclass,
                    DummySecondSubclass,
                    DummyMixedSubclass,
                    DummyLeafClass,
                }
                self.assertEqual(
                    set(
                        get_all_subclasses(
                            DummyRootClass, include_cls_self=include_cls_self
                        )
                    ),
                    {DummyRootClass} | subclasses if include_cls_self else subclasses,
                )
            with self.subTest(
                "Test with the second subclass (parents should not be included)",
                include_cls_self=include_cls_self,
            ):
                subclasses = {DummyMixedSubclass, DummyLeafClass}
                self.assertEqual(
                    set(
                        get_all_subclasses(
                            DummySecondSubclass, include_cls_self=include_cls_self
                        )
                    ),
                    {DummySecondSubclass} | subclasses
                    if include_cls_self
                    else subclasses,
                )
            with self.subTest(
                "Test with leaf class (no subclasses)",
                include_cls_self=include_cls_self,
            ):
                self.assertEqual(
                    get_all_subclasses(
                        DummyLeafClass, include_cls_self=include_cls_self
                    ),
                    [DummyLeafClass] if include_cls_self else [],
                )

"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import re
import unittest

from commons import get_all_subclasses

from matrix_functions_types import QREigendecompositionConfig


class QREigendecompositionConfigSubclassesTest(unittest.TestCase):
    def test_illegal_tolerance(self) -> None:
        for cls in get_all_subclasses(QREigendecompositionConfig):
            # tolerance has to be in the interval [0.0, 1.0].
            for tolerance in [-1.0, 1.1]:
                with self.subTest(cls=cls):
                    self.assertRaisesRegex(
                        ValueError,
                        re.escape(
                            f"Invalid tolerance value: {tolerance}. Must be in the interval [0.0, 1.0]."
                        ),
                        cls,
                        tolerance=tolerance,
                    )

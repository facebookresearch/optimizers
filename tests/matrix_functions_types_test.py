import re
import unittest

from distributed_shampoo.tests.shampoo_types_test import get_all_subclasses

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

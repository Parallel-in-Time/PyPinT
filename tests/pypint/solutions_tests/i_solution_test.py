# coding=utf-8
import unittest

from pypint.solutions.i_solution import ISolution


class ISolutionTest(unittest.TestCase):
    def setUp(self):
        self._default = ISolution()

    def test_provides_used_iterations(self):
        self.assertEqual(self._default.used_iterations, 0)
        self._default.used_iterations = 2
        self.assertEqual(self._default.used_iterations, 2)

        with self.assertRaises(ValueError):
            self._default.used_iterations = -2

    def test_defines_add_solution_interface(self):
        self.assertRaises(NotImplementedError, self._default.add_solution)

    def test_provides_data_storage_type(self):
        self.assertIsNone(self._default.data_storage_type)

    def test_has_to_string_method(self):
        self.assertRegex(self._default.__str__(), "ISolution")

if __name__ == "__main__":
    unittest.main()

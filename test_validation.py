import unittest
from greeting import greeting

class ValidationTest(unittest.TestCase):
    def test_trim(self):
        self.assertEqual(greeting(" Ada "), "Hello, Ada!")
    def test_blank(self):
        with self.assertRaises(ValueError):
            greeting("  ")

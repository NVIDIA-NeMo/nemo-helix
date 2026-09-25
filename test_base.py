import unittest
from greeting import greeting

class BaseTest(unittest.TestCase):
    def test_default(self):
        self.assertEqual(greeting("Ada"), "Hello, Ada!")

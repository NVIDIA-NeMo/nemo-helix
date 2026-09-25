import unittest
from greeting import greeting

class PrefixTest(unittest.TestCase):
    def test_custom(self):
        self.assertEqual(greeting("Ada", prefix="Welcome"), "Welcome, Ada!")

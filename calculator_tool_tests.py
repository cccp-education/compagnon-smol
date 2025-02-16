# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase


class CalculatorToolTests(TestCase):

    @staticmethod
    def test_canary_calculator():
        print("canary_calculator")
        # assert_that(2 + 2).is_equal_to(4)

    # @staticmethod
    def test_bird_calculator(self):
        print("bird_calculator")
        # assert_that(2 + 2).is_equal_to(4)


if __name__ == '__main__':
    unittest.main()

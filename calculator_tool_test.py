# -*- coding: utf-8 -*-

from assertpy import assert_that

from calculator_tool import multiply, plus


class TestCalculatorTool:

    @staticmethod
    def test_multiply():
        assert_that(multiply(2, 2)).is_equal_to(4)

    @staticmethod
    def test_plus():
        assert_that(plus(2, 2)).is_equal_to(4)

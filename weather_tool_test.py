# -*- coding: utf-8 -*-

from assertpy import assert_that


class TestWeatherTool:

    @staticmethod
    def canary_weather():
        assert_that(2 + 2).is_equal_to(4)

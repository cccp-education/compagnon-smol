# -*- coding: utf-8 -*-

import os
import unittest
from unittest import TestCase

from assertpy import assert_that

from utils import clear_environment, ENV, set_environment


class UtilsTests(TestCase):

    @staticmethod
    def test_clear_environment():
        assert_that(os.environ).is_not_empty()
        assert_that(os.environ.items()).is_not_empty()
        assert_that("foo").is_not_in(os.environ)
        assert_that("bar").is_not_in(os.environ.values())
        os.environ["foo"] = "bar"
        assert_that(os.environ).contains_key("foo")
        assert_that(os.environ).contains_value("bar")
        clear_environment({"foo": "bar"})
        assert_that(os.environ).is_not_empty()
        assert_that(os.environ.keys()).does_not_contain("foo")
        assert_that(os.environ.values()).does_not_contain("bar")

    @staticmethod
    def test_set_environment_loop_impl():
        for key, value in ENV.items():
            assert_that(os.environ).does_not_contain(key)
        set_environment()
        for key, value in ENV.items():
            assert_that(os.environ).contains_key(key)
            assert_that(os.environ[key]).is_equal_to(value)
        clear_environment(ENV)

    @staticmethod
    def test_set_environment_map_impl():
        assert_that(list(map(
            lambda key: key in os.environ, ENV.keys()
        ))).is_equal_to([False] * len(ENV))
        set_environment()
        assert_that(list(map(
            lambda key: os.environ[key] == ENV[key], ENV.keys()
        ))).is_equal_to([True] * len(ENV))
        clear_environment(ENV)


if __name__ == '__main__':
    unittest.main()

import sys
import unittest
from cqcpy.tests import test_spin_utils

def full_suite():
    suite = unittest.TestSuite()

    suite.addTest(test_spin_utils.SpinUtilsTest("test_F_sym"))
    suite.addTest(test_spin_utils.SpinUtilsTest("test_I_sym"))
    suite.addTest(test_spin_utils.SpinUtilsTest("test_I"))
    #suite.addTest(test_spin_utils.SpinUtilsTest("test_Be_plus"))
    suite.addTest(test_spin_utils.SpinUtilsTest("test_T"))

    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(full_suite())

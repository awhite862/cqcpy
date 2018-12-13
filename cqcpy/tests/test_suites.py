import sys
import unittest
from cqcpy.tests import test_spin_utils
from cqcpy.tests import test_test

def full_suite():
    suite = unittest.TestSuite()

    suite.addTest(test_spin_utils.SpinUtilsTest("test_F_sym"))
    suite.addTest(test_spin_utils.SpinUtilsTest("test_I_sym"))
    suite.addTest(test_spin_utils.SpinUtilsTest("test_I"))
    #suite.addTest(test_spin_utils.SpinUtilsTest("test_Be_plus"))
    suite.addTest(test_spin_utils.SpinUtilsTest("test_T"))

    suite.addTest(test_test.TestTest("test_framework"))
    suite.addTest(test_test.TestTest("test_Isym"))
    suite.addTest(test_test.TestTest("test_int_sym"))
    suite.addTest(test_test.TestTest("test_Tsym"))
    suite.addTest(test_test.TestTest("test_Lsym"))
    suite.addTest(test_test.TestTest("test_ft_int_sym"))
    suite.addTest(test_test.TestTest("test_ft_Tsym"))

    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(full_suite())

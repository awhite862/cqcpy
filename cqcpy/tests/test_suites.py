import unittest
from cqcpy.tests import test_cc_ampl
from cqcpy.tests import test_cc_rdm
from cqcpy.tests import test_ci_utils
from cqcpy.tests import test_ft_utils
from cqcpy.tests import test_integrals
from cqcpy.tests import test_lambda_equations
from cqcpy.tests import test_spin_utils
from cqcpy.tests import test_test
from cqcpy.tests import test_utils


def full_suite():
    try:
        import pyscf
        with_pyscf = True
    except ImportError:
        with_pyscf = False

    suite = unittest.TestSuite()

    suite.addTest(test_cc_ampl.TamplEquationsTest("test_ccsd_stanton"))
    suite.addTest(test_cc_ampl.TamplEquationsTest("test_ccd"))
    suite.addTest(test_cc_ampl.TamplEquationsTest("test_ccd_stanton"))
    suite.addTest(test_cc_ampl.TamplEquationsTest("test_ucc_energy"))
    suite.addTest(test_cc_ampl.TamplEquationsTest("test_uccsd"))
    suite.addTest(test_cc_ampl.TamplEquationsTest("test_rcc_energy"))
    suite.addTest(test_cc_ampl.TamplEquationsTest("test_rccsd"))

    suite.addTest(test_cc_rdm.CCRDMTest("test_1rdm_opt"))
    suite.addTest(test_cc_rdm.CCRDMTest("test_2rdm_opt"))
    suite.addTest(test_cc_rdm.CCRDMTest("test_u1rdm"))
    suite.addTest(test_cc_rdm.CCRDMTest("test_u2rdm"))
    suite.addTest(test_cc_rdm.CCRDMTest("test_r1rdm"))
    suite.addTest(test_cc_rdm.CCRDMTest("test_r2rdm"))

    suite.addTest(test_ci_utils.CIUtilsTest("test_mat_on_vec_cisd"))
    suite.addTest(test_ci_utils.CIUtilsTest("test_cis"))
    suite.addTest(test_ci_utils.CIUtilsTest("test_cisd"))

    suite.addTest(test_ft_utils.FTUtilsTest("test_fermi"))
    suite.addTest(test_ft_utils.FTUtilsTest("test_vfermi"))
    suite.addTest(test_ft_utils.FTUtilsTest("test_dfermi"))
    suite.addTest(test_ft_utils.FTUtilsTest("test_vector"))
    suite.addTest(test_ft_utils.FTUtilsTest("test_dgrand_potentiala"))
    suite.addTest(test_ft_utils.FTUtilsTest("test_dgrand_potentialb"))
    suite.addTest(test_ft_utils.FTUtilsTest("test_dgrand_potentialc"))

    suite.addTest(
        test_lambda_equations.LambdaEquationsTest("test_ccsd_opt"))
    suite.addTest(
        test_lambda_equations.LambdaEquationsTest("test_ccsd_stanton"))
    suite.addTest(
        test_lambda_equations.LambdaEquationsTest("test_ccsd_opt_int"))
    suite.addTest(
        test_lambda_equations.LambdaEquationsTest("test_ccd"))
    suite.addTest(
        test_lambda_equations.LambdaEquationsTest("test_uccsd_lambda"))
    suite.addTest(
        test_lambda_equations.LambdaEquationsTest("test_rccsd_lambda"))

    suite.addTest(test_spin_utils.SpinUtilsTest("test_F_sym"))
    suite.addTest(test_spin_utils.SpinUtilsTest("test_I_sym"))
    suite.addTest(test_spin_utils.SpinUtilsTest("test_T"))

    suite.addTest(test_test.TestTest("test_framework"))
    suite.addTest(test_test.TestTest("test_Isym"))
    suite.addTest(test_test.TestTest("test_int_sym"))
    suite.addTest(test_test.TestTest("test_Tsym"))
    suite.addTest(test_test.TestTest("test_Lsym"))
    suite.addTest(test_test.TestTest("test_ft_D_sym"))
    suite.addTest(test_test.TestTest("test_ft_int_sym"))
    suite.addTest(test_test.TestTest("test_ft_Tsym"))

    suite.addTest(test_utils.UtilsTest("test_block_diag"))
    suite.addTest(test_utils.UtilsTest("test_D1"))
    suite.addTest(test_utils.UtilsTest("test_D2"))

    if not with_pyscf:
        print("WARNING: PySCF not found, skipping some tests")
    else:
        suite.addTest(test_spin_utils.SpinUtilsTest("test_Be_plus"))
        suite.addTest(test_spin_utils.SpinUtilsTest("test_I"))

        suite.addTest(test_integrals.IntegralsTest("test_phys"))
        suite.addTest(test_integrals.IntegralsTest("test_u"))
        suite.addTest(test_integrals.IntegralsTest("test_sol_phys"))
        suite.addTest(test_integrals.IntegralsTest("test_fock"))
        suite.addTest(test_integrals.IntegralsTest("test_fock_sol"))
        suite.addTest(test_integrals.IntegralsTest("test_fock_sol_k"))
        suite.addTest(test_integrals.IntegralsTest("test_eri_sol_k"))

    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(full_suite())

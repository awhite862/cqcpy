import unittest
import numpy

from cqcpy import test_utils
import cqcpy.spin_utils as spin_utils
import cqcpy.cc_equations as cc_equations

class CCRDMTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_1rdm_opt(self):
        no = 4
        nv = 8
        thresh = 1e-12
        T1,T2 = test_utils.make_random_T(no,nv)
        L1,L2 = test_utils.make_random_L(no,nv)

        pba_ref = cc_equations.ccsd_1rdm_ba(T1,T2,L1,L2)
        pba_out = cc_equations.ccsd_1rdm_ba_opt(T1,T2,L1,L2)
        diff = numpy.linalg.norm(pba_ref - pba_out)/numpy.sqrt(pba_ref.size)
        self.assertTrue(diff < thresh,"Error in p_ba: {}".format(diff))

        pji_ref = cc_equations.ccsd_1rdm_ji(T1,T2,L1,L2)
        pji_out = cc_equations.ccsd_1rdm_ji_opt(T1,T2,L1,L2)
        diff = numpy.linalg.norm(pji_ref - pji_out)/numpy.sqrt(pji_ref.size)
        self.assertTrue(diff < thresh,"Error in p_ji: {}".format(diff))

        pai_ref = cc_equations.ccsd_1rdm_ai(T1,T2,L1,L2)
        pai_out = cc_equations.ccsd_1rdm_ai_opt(T1,T2,L1,L2)
        diff = numpy.linalg.norm(pai_ref - pai_out)/numpy.sqrt(pai_ref.size)
        self.assertTrue(diff < thresh,"Error in p_ai: {}".format(diff))

    def test_2rdm_opt(self):
        no = 4
        nv = 8
        thresh = 1e-12
        T1,T2 = test_utils.make_random_T(no,nv)
        L1,L2 = test_utils.make_random_L(no,nv)

        pcdab_ref = cc_equations.ccsd_2rdm_cdab(T1,T2,L1,L2)
        pcdab_out = cc_equations.ccsd_2rdm_cdab_opt(T1,T2,L1,L2)
        diff = numpy.linalg.norm(pcdab_ref - pcdab_out)/numpy.sqrt(pcdab_ref.size)
        self.assertTrue(diff < thresh,"Error in p_cdab: {}".format(diff))

        pbcai_ref = cc_equations.ccsd_2rdm_bcai(T1,T2,L1,L2)
        pbcai_out = cc_equations.ccsd_2rdm_bcai_opt(T1,T2,L1,L2)
        diff = numpy.linalg.norm(pbcai_ref - pbcai_out)/numpy.sqrt(pbcai_ref.size)
        self.assertTrue(diff < thresh,"Error in p_bcai: {}".format(diff))

        pbjai_ref = cc_equations.ccsd_2rdm_bjai(T1,T2,L1,L2)
        pbjai_out = cc_equations.ccsd_2rdm_bjai_opt(T1,T2,L1,L2)
        diff = numpy.linalg.norm(pbjai_ref - pbjai_out)/numpy.sqrt(pbjai_ref.size)
        self.assertTrue(diff < thresh,"Error in p_bjai: {}".format(diff))

        pabij_ref = cc_equations.ccsd_2rdm_abij(T1,T2,L1,L2)
        pabij_out = cc_equations.ccsd_2rdm_abij_opt(T1,T2,L1,L2)
        diff = numpy.linalg.norm(pabij_ref - pabij_out)/numpy.sqrt(pabij_ref.size)
        self.assertTrue(diff < thresh,"Error in p_abij: {}".format(diff))

        pkaij_ref = cc_equations.ccsd_2rdm_kaij(T1,T2,L1,L2)
        pkaij_out = cc_equations.ccsd_2rdm_kaij_opt(T1,T2,L1,L2)
        diff = numpy.linalg.norm(pkaij_ref - pkaij_out)/numpy.sqrt(pkaij_ref.size)
        self.assertTrue(diff < thresh,"Error in p_kaij: {}".format(diff))

        pklij_ref = cc_equations.ccsd_2rdm_klij(T1,T2,L1,L2)
        pklij_out = cc_equations.ccsd_2rdm_klij_opt(T1,T2,L1,L2)
        diff = numpy.linalg.norm(pklij_ref - pklij_out)/numpy.sqrt(pklij_ref.size)
        self.assertTrue(diff < thresh,"Error in p_klij: {}".format(diff))

    #def test_u1rdm(self):
    #    pass

if __name__ == '__main__':
    unittest.main()

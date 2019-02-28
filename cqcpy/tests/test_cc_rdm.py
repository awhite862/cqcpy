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
        #F,I = test_utils.make_random_integrals(no,nv)
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

    #def test_u1rdm(self):
    #    pass

if __name__ == '__main__':
    unittest.main()

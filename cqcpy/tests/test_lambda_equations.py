import unittest
import numpy
from cqcpy import test_utils
import cqcpy.spin_utils as spin_utils
import cqcpy.cc_equations as cc_equations
import cqcpy.ov_blocks as ov_blocks

class LambdaEquationsTest(unittest.TestCase):
    def setUp(self):
        self.thresh = 1e-12
        self.no = 3
        self.nv = 5

    def test_ccsd_opt(self):
        no = self.no
        nv = self.nv
        T1old,T2old = test_utils.make_random_T(no,nv)
        L1old,L2old = test_utils.make_random_L(no,nv)
        F,I = test_utils.make_random_integrals(no,nv)

        L1sim, L2sim = cc_equations.ccsd_lambda_simple(F, I, L1old, L2old, T1old, T2old)
        L1opt, L2opt = cc_equations.ccsd_lambda_opt(F, I, L1old, L2old, T1old, T2old)

        D1 = numpy.linalg.norm(L1sim - L1opt)
        D2 = numpy.linalg.norm(L2sim - L2opt)
        s1 = D1 < self.thresh
        s2 = D2 < self.thresh
        e1 = "Error in optimized L1"
        e2 = "Error in optimized L2"
        self.assertTrue(s1,e1)
        self.assertTrue(s2,e2)

    def test_ccsd_stanton(self):
        no = self.no
        nv = self.nv
        T1old,T2old = test_utils.make_random_T(no,nv)
        L1old,L2old = test_utils.make_random_L(no,nv)
        F,I = test_utils.make_random_integrals(no,nv)

        L1sim, L2sim = cc_equations.ccsd_lambda_simple(F, I, L1old, L2old, T1old, T2old)
        L1opt, L2opt = cc_equations.ccsd_lambda_stanton(F, I, L1old, L2old, T1old, T2old)

        D1 = numpy.linalg.norm(L1sim - L1opt)
        D2 = numpy.linalg.norm(L2sim - L2opt)
        s1 = D1 < self.thresh
        s2 = D2 < self.thresh
        e1 = "Error in optimized L1"
        e2 = "Error in optimized L2"
        self.assertTrue(s1,e1)
        self.assertTrue(s2,e2)

    def test_ccsd_opt_int(self):
        no = self.no
        nv = self.nv
        T1old,T2old = test_utils.make_random_T(no,nv)
        L1old,L2old = test_utils.make_random_L(no,nv)
        F,I = test_utils.make_random_integrals(no,nv)
        intor = cc_equations.lambda_int(F,I,T1old,T2old)

        L1sim, L2sim = cc_equations.ccsd_lambda_simple(F, I, L1old, L2old, T1old, T2old)
        L1opt, L2opt = cc_equations.ccsd_lambda_opt_int(F, I, L1old, L2old, T1old, T2old, intor)

        D1 = numpy.linalg.norm(L1sim - L1opt)
        D2 = numpy.linalg.norm(L2sim - L2opt)
        s1 = D1 < self.thresh
        s2 = D2 < self.thresh
        e1 = "Error in optimized L1"
        e2 = "Error in optimized L2"
        self.assertTrue(s1,e1)
        self.assertTrue(s2,e2)

    def test_ccd(self):
        no = self.no
        nv = self.nv
        T1old,T2old = test_utils.make_random_T(no,nv)
        L1old,L2old = test_utils.make_random_L(no,nv)
        F,I = test_utils.make_random_integrals(no,nv)
        T1old = numpy.zeros((nv,no))
        L1old = numpy.zeros((no,nv))

        L2 = cc_equations.ccd_lambda_simple(F, I, L2old, T2old)
        L1t,L2t = cc_equations.ccsd_lambda_simple(F, I, L1old, L2old, T1old, T2old)
        D = numpy.linalg.norm(L2 - L2t)
        s = D < self.thresh
        err = "Error in CCD L2"
        self.assertTrue(s,err)

    def test_uccsd_lambda(self):
        noa = self.no
        nob = self.no
        nva = self.nv
        nvb = self.nv
        Fa = test_utils.make_random_F(noa, nva)
        Fb = test_utils.make_random_F(nob, nvb)

        # Direct integrals over a,b orbitals
        Ia = test_utils.make_random_I_anti(noa,nva)
        Ib = test_utils.make_random_I_anti(nob,nvb)
        Iabab = test_utils.make_random_Ifull_gen(
            noa,nva,nob,nvb,noa,nva,nob,nvb)

        # Full antisymmetric spin-orbital tensor
        I = spin_utils.int_to_spin2(Ia, Ib, Iabab, noa, nva, nob, nvb)
        F = spin_utils.F_to_spin(Fa, Fb, noa, nva, nob, nvb)

        # initial T
        T1a,T1b = test_utils.make_random_T1_spatial(noa,nva,nob,nvb)
        T2aa,T2ab,T2bb = test_utils.make_random_T2_spatial(noa,nva,nob,nvb)
        T1 = spin_utils.T1_to_spin(T1a,T1b,noa,nva,nob,nvb)
        T2 = spin_utils.T2_to_spin(T2aa,T2ab,T2bb,noa,nva,nob,nvb)

        # initial L
        L1aold,L1bold = test_utils.make_random_T1_spatial(nva,noa,nvb,nob)
        L2aaold,L2abold,L2bbold = test_utils.make_random_T2_spatial(nva,noa,nvb,nob)
        L1old = spin_utils.T1_to_spin(L1aold,L1bold,nva,noa,nvb,nob)
        L2old = spin_utils.T2_to_spin(L2aaold,L2abold,L2bbold,nva,noa,nvb,nob)

        # Get updated Lambda using spin orbitals
        L1_ref,L2_ref = cc_equations.ccsd_lambda_opt(F, I, L1old, L2old, T1, T2)

        # Get updated Lambda using unrestricted code
        M1,M2 = cc_equations.uccsd_lambda_opt(Fa, Fb, Ia, Ib, Iabab,
            (L1aold,L1bold), (L2aaold, L2abold, L2bbold), (T1a,T1b),(T2aa, T2ab, T2bb))

        L1a,L1b = M1
        L2aa,L2ab,L2bb = M2
        L1 = spin_utils.T1_to_spin(L1a, L1b, nva, noa, nvb, nob)
        L2 = spin_utils.T2_to_spin(L2aa, L2ab, L2bb, nva, noa, nvb, nob)
        z1 = numpy.linalg.norm(L1 - L1_ref) / numpy.sqrt(L1.size)
        z2 = numpy.linalg.norm(L2 - L2_ref) / numpy.sqrt(L2.size)
        s1 = z1 < self.thresh
        s2 = z2 < self.thresh
        e1 = "Error in UCCSD L1"
        e2 = "Error in UCCSD L2"
        self.assertTrue(s1,e1)
        self.assertTrue(s2,e2)

    def test_rccsd_lambda(self):
        no = 3
        nv = 5
        n = no + nv
        F = test_utils.make_random_F(no, nv)
        Itot = numpy.random.random((n,n,n,n))
        Itot = Itot + Itot.transpose((1,0,3,2))
        Ianti = Itot - Itot.transpose((0,1,3,2))
        I = ov_blocks.make_two_e_blocks_full(Itot,no,nv,no,nv,no,nv,no,nv)
        Ia = ov_blocks.make_two_e_blocks(Ianti, no, nv, no, nv, no, nv, no, nv)

        T1 = numpy.random.random((nv,no))
        T2 = numpy.random.random((nv,nv,no,no))
        T2 = T2 + T2.transpose((1,0,3,2))

        L1 = numpy.random.random((no,nv))
        L2 = numpy.random.random((no,no,nv,nv))
        L2 = L2 + L2.transpose((1,0,3,2))

        T1a = T1b = T1
        T2aa = T2 - T2.transpose((0,1,3,2))

        L2aa = L2 - L2.transpose((0,1,3,2))

        # Update with UCCSD
        M1,M2 = cc_equations.uccsd_lambda_opt(F, F, Ia, Ia, I, (L1,L1),
                (L2aa, L2, L2aa), (T1a,T1b),(T2aa, T2, T2aa))
        rL1,rL2 = cc_equations.rccsd_lambda_opt(F, I, L1, L2, T1, T2)
        ref1 = M1[0]
        ref2 = M2[1]

        d1 = numpy.linalg.norm(ref1 - rL1) / numpy.linalg.norm(ref1)
        d2 = numpy.linalg.norm(ref2 - rL2) / numpy.linalg.norm(ref2)

        e1 = "Error in RCCSD L1"
        e2 = "Error in RCCSD L2"
        self.assertTrue(d1 < 1e-14,e1)
        self.assertTrue(d2 < 1e-14,e2)

if __name__ == '__main__':
    unittest.main()

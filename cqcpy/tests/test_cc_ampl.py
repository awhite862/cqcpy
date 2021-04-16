import unittest
import numpy

from cqcpy import test_utils
from cqcpy import ov_blocks
import cqcpy.spin_utils as spin_utils
import cqcpy.cc_energy as cc_energy
import cqcpy.cc_equations as cc_equations


class TamplEquationsTest(unittest.TestCase):
    def setUp(self):
        self.thresh = 1e-12
        self.no = 3
        self.nv = 5

    def test_ccsd_stanton(self):
        no = self.no
        nv = self.nv
        T1old, T2old = test_utils.make_random_T(no, nv)
        F, I = test_utils.make_random_integrals(no, nv)

        T1sim, T2sim = cc_equations.ccsd_simple(F, I, T1old, T2old)
        T1stn, T2stn = cc_equations.ccsd_stanton(F, I, T1old, T2old)

        D1 = numpy.linalg.norm(T1sim - T1stn)
        D2 = numpy.linalg.norm(T2sim - T2stn)
        s1 = D1 < self.thresh
        s2 = D2 < self.thresh
        e1 = "Error in optimized T1"
        e2 = "Error in optimized T2"
        self.assertTrue(s1, e1)
        self.assertTrue(s2, e2)

    def test_ccd(self):
        no = self.no
        nv = self.nv
        T1old, T2old = test_utils.make_random_T(no, nv)
        F, I = test_utils.make_random_integrals(no, nv)
        T1old = numpy.zeros((nv,no))

        T2 = cc_equations.ccd_simple(F, I, T2old)
        T1sd, T2sd = cc_equations.ccsd_simple(F, I, T1old, T2old)

        D = numpy.linalg.norm(T2 - T2sd)
        s = D < self.thresh
        err = "Error in CCD T2"
        self.assertTrue(s, err)

    def test_ccd_stanton(self):
        no = self.no
        nv = self.nv
        T1old, T2old = test_utils.make_random_T(no, nv)
        F, I = test_utils.make_random_integrals(no, nv)

        T2 = cc_equations.ccd_simple(F, I, T2old)
        T2sd = cc_equations.ccd_stanton(F, I, T2old)

        D = numpy.linalg.norm(T2 - T2sd)
        s = D < self.thresh
        err = "Error in CCD T2"
        self.assertTrue(s, err)

    def test_ucc_energy(self):
        noa = self.no
        nob = self.no
        nva = self.nv
        nvb = self.nv
        Faa = test_utils.make_random_F(noa, nva)
        Fbb = test_utils.make_random_F(nob, nvb)

        # Direct integrals over a,b orbitals
        Ia = test_utils.make_random_I_anti(noa, nva)
        Ib = test_utils.make_random_I_anti(nob, nvb)
        Iabab = test_utils.make_random_Ifull_gen(
            noa, nva, nob, nvb, noa, nva, nob, nvb)

        # Full antisymmetric spin-orbital tensor
        I = spin_utils.int_to_spin2(Ia, Ib, Iabab, noa, nva, nob, nvb)
        F = spin_utils.F_to_spin(Faa, Fbb, noa, nva, nob, nvb)

        # initial T
        T1a, T1b = test_utils.make_random_T1_spatial(noa, nva, nob, nvb)
        T2aa, T2ab, T2bb = test_utils.make_random_T2_spatial(noa, nva, nob, nvb)
        T1 = spin_utils.T1_to_spin(T1a, T1b, noa, nva, nob, nvb)
        T2 = spin_utils.T2_to_spin(T2aa, T2ab, T2bb, noa, nva, nob, nvb)

        E_ref = cc_energy.cc_energy(T1, T2, F.ov, I.oovv)
        E_out = cc_energy.ucc_energy(
            (T1a,T1b), (T2aa,T2ab,T2bb),
            Faa.ov, Fbb.ov, Ia.oovv, Ib.oovv, Iabab.oovv)
        s = abs(E_ref - E_out) < self.thresh
        err = "Error in ucc_energy"
        self.assertTrue(s, err)

    def test_uccsd(self):
        noa = self.no
        nob = self.no
        nva = self.nv
        nvb = self.nv
        Faa = test_utils.make_random_F(noa, nva)
        Fbb = test_utils.make_random_F(nob, nvb)

        # Direct integrals over a,b orbitals
        Ia = test_utils.make_random_I_anti(noa, nva)
        Ib = test_utils.make_random_I_anti(nob, nvb)
        I_abab = test_utils.make_random_Ifull_gen(
            noa, nva, nob, nvb, noa, nva, nob, nvb)

        # Full antisymmetric spin-orbital tensor
        I = spin_utils.int_to_spin2(Ia, Ib, I_abab, noa, nva, nob, nvb)
        F = spin_utils.F_to_spin(Faa, Fbb, noa, nva, nob, nvb)

        # initial T
        T1a, T1b = test_utils.make_random_T1_spatial(noa, nva, nob, nvb)
        T2aa, T2ab, T2bb = test_utils.make_random_T2_spatial(noa, nva, nob, nvb)
        T1 = spin_utils.T1_to_spin(T1a, T1b, noa, nva, nob, nvb)
        T2 = spin_utils.T2_to_spin(T2aa, T2ab, T2bb, noa, nva, nob, nvb)

        # Update with spin orbitals
        S1ref, S2ref = cc_equations.ccsd_stanton(F, I, T1, T2)

        # Update with UCCSD
        S1, S2 = cc_equations.uccsd_stanton(
            Faa, Fbb, Ia, Ib, I_abab,
            (T1a,T1b), (T2aa,T2ab,T2bb))
        S1a, S1b = S1
        S2aa, S2ab, S2bb = S2
        S1 = spin_utils.T1_to_spin(S1a, S1b, noa, nva, nob, nvb)
        S2 = spin_utils.T2_to_spin(S2aa, S2ab, S2bb, noa, nva, nob, nvb)
        z1 = numpy.linalg.norm(S1 - S1ref) / numpy.sqrt(S1.size)
        z2 = numpy.linalg.norm(S2 - S2ref) / numpy.sqrt(S2.size)
        s1 = z1 < self.thresh
        s2 = z2 < self.thresh
        e1 = "Error in UCCSD T1"
        e2 = "Error in UCCSD T2"
        self.assertTrue(s1, e1)
        self.assertTrue(s2, e2)

    def test_rcc_energy(self):
        no = 3
        nv = 5
        n = no + nv
        F = test_utils.make_random_F(no, nv)
        Itot = numpy.random.random((n,n,n,n))
        Itot = Itot + Itot.transpose((1,0,3,2))
        I = ov_blocks.make_two_e_blocks_full(
            Itot, no, nv, no, nv, no, nv, no, nv)
        Ianti = I.oovv - I.oovv.transpose((0,1,3,2))

        T1 = numpy.random.random((nv,no))
        T2 = numpy.random.random((nv,nv,no,no))
        T2 = T2 + T2.transpose((1,0,3,2))
        T2aa = T2 - T2.transpose((0,1,3,2))

        E_ref = cc_energy.ucc_energy(
            (T1,T1), (T2aa,T2,T2aa), F.ov, F.ov, Ianti, Ianti, I.oovv)
        E_out = cc_energy.rcc_energy(T1, T2, F.ov, I.oovv)

        s = abs(E_ref - E_out) < self.thresh
        err = "Error in rcc_energy"
        self.assertTrue(s, err)

    def test_rccsd(self):
        no = 3
        nv = 5
        n = no + nv
        F = test_utils.make_random_F(no, nv)
        Itot = numpy.random.random((n,n,n,n))
        Itot = Itot + Itot.transpose((1,0,3,2))
        Ianti = Itot - Itot.transpose((0,1,3,2))
        I = ov_blocks.make_two_e_blocks_full(
            Itot, no, nv, no, nv, no, nv, no, nv)
        Ia = ov_blocks.make_two_e_blocks(
            Ianti, no, nv, no, nv, no, nv, no, nv)

        T1 = numpy.random.random((nv,no))
        T2 = numpy.random.random((nv,nv,no,no))
        T2 = T2 + T2.transpose((1,0,3,2))
        T1a = T1b = T1
        T2aa = T2 - T2.transpose((0,1,3,2))

        # Update with UCCSD
        uS1, uS2 = cc_equations.uccsd_stanton(
            F, F, Ia, Ia, I,
            (T1a,T1b), (T2aa,T2,T2aa))
        ref1 = uS1[0]
        ref2 = uS2[1]
        rS1, rS2 = cc_equations.rccsd_stanton(F, I, T1, T2)

        d1 = numpy.linalg.norm(ref1 - rS1) / numpy.linalg.norm(ref1)
        d2 = numpy.linalg.norm(ref2 - rS2) / numpy.linalg.norm(ref2)

        e1 = "Error in RCCSD T1"
        e2 = "Error in RCCSD T2"
        self.assertTrue(d1 < 1e-14, e1)
        self.assertTrue(d2 < 1e-14, e2)


if __name__ == '__main__':
    unittest.main()

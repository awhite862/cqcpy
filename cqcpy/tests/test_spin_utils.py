import unittest
import numpy

from cqcpy import test_utils
import cqcpy.spin_utils as spin_utils


class SpinUtilsTest(unittest.TestCase):
    def setUp(self):
        self.thresh = 1e-14

    def test_F_sym(self):
        noa = 2
        nva = 3
        nob = 2
        nvb = 3
        Faa = test_utils.make_random_F(noa, nva)
        Fbb = test_utils.make_random_F(nob, nvb)
        F = spin_utils.F_to_spin(Faa, Fbb, noa, nva, nob, nvb)
        z = F.oo[:noa,noa:]
        s = numpy.linalg.norm(z) < self.thresh
        err = "non-zero ab block of Foo"
        self.assertTrue(s,err)

        z = F.ov[:noa,nva:]
        s = numpy.linalg.norm(z) < self.thresh
        err = "non-zero ab block of Fov"
        self.assertTrue(s,err)

        z = F.vo[:nva,noa:]
        s = numpy.linalg.norm(z) < self.thresh
        err = "non-zero ab block of Fvo"
        self.assertTrue(s,err)

        z = F.vo[:nva,nva:]
        s = numpy.linalg.norm(z) < self.thresh
        err = "non-zero ab block of Fvv"
        self.assertTrue(s,err)

    def test_I_sym(self):
        noa = 3
        nob = 2
        nva = 2
        nvb = 3
        na = noa + nva
        nb = nob + nvb
        no = noa + nob
        nv = nva + nvb

        # random integrals over a,b spatial orbitals
        I_aaaa = test_utils.make_random_Ifull(noa,nva)
        I_bbbb = test_utils.make_random_Ifull(nob,nvb)
        I_abab = test_utils.make_random_Ifull_gen(
            noa,nva,nob,nvb,noa,nva,nob,nvb)

        I = spin_utils.int_to_spin(I_aaaa, I_bbbb, I_abab, noa, nva, nob, nvb)

        # check a coupled selected blocks that should be zero
        z = I.oooo[noa:,:noa,:noa,:noa]
        s = numpy.linalg.norm(z) < self.thresh
        err = "non-zero abbb block of Ioooo"
        self.assertTrue(s,err)
        z = I.ooov[noa:,:noa,:noa,:nva]
        s = numpy.linalg.norm(z) < self.thresh
        err = "non-zero abbb block of Iooov"
        self.assertTrue(s,err)
        z = I.vovo[nva:,:noa,:nva,:noa]
        s = numpy.linalg.norm(z) < self.thresh
        err = "non-zero abbb block of Ivovo"
        self.assertTrue(s,err)
        z = I.vvvv[nva:,:nva,:nva,:nva]
        s = numpy.linalg.norm(z) < self.thresh
        err = "non-zero abbb block of Ivvvv"
        self.assertTrue(s,err)

    def test_I(self):
        noa = 3
        nob = 2
        nva = 2
        nvb = 3
        na = noa + nva
        nb = nob + nvb
        no = noa + nob
        nv = nva + nvb

        # random integrals over a,b spatial orbitals
        Ia_ref = test_utils.make_random_I(noa,nva)
        Ib_ref = test_utils.make_random_I(nob,nvb)
        Iabab_ref = test_utils.make_random_Ifull_gen(
            noa,nva,nob,nvb,noa,nva,nob,nvb)

        I = spin_utils.int_to_spin2(Ia_ref, Ib_ref, Iabab_ref, noa, nva, nob, nvb)
        Ia,Ib,Iabab = spin_utils.int_to_spatial(I, noa, nob, nva, nvb)

        test = Ia.vvvv - Ia_ref.vvvv
        s = numpy.linalg.norm(test) < self.thresh
        err = "error in Ia vvvv integrals"
        self.assertTrue(s,err)

        test = Ia.vvvo - Ia_ref.vvvo
        s = numpy.linalg.norm(test) < self.thresh
        err = "error in Ia vvvo integrals"
        self.assertTrue(s,err)

        test = Ia.vovv - Ia_ref.vovv
        s = numpy.linalg.norm(test) < self.thresh
        err = "error in Ia vovv integrals"
        self.assertTrue(s,err)

        test = Ia.vvoo - Ia_ref.vvoo
        s = numpy.linalg.norm(test) < self.thresh
        err = "error in Ia vvoo integrals"
        self.assertTrue(s,err)

        test = Ia.vovo - Ia_ref.vovo
        s = numpy.linalg.norm(test) < self.thresh
        err = "error in Ia vovo integrals"
        self.assertTrue(s,err)

        test = Ia.oovv - Ia_ref.oovv
        s = numpy.linalg.norm(test) < self.thresh
        err = "error in Ia oovv integrals"
        self.assertTrue(s,err)

        test = Ia.vooo - Ia_ref.vooo
        s = numpy.linalg.norm(test) < self.thresh
        err = "error in Ia vooo integrals"
        self.assertTrue(s,err)

        test = Ia.ooov - Ia_ref.ooov
        s = numpy.linalg.norm(test) < self.thresh
        err = "error in Ia ooov integrals"
        self.assertTrue(s,err)

        test = Ia.oooo - Ia_ref.oooo
        s = numpy.linalg.norm(test) < self.thresh
        err = "error in Ia oooo integrals"
        self.assertTrue(s,err)

        test = Ib.vvvv - Ib_ref.vvvv
        s = numpy.linalg.norm(test) < self.thresh
        err = "error in Ib vvvv integrals"
        self.assertTrue(s,err)

        test = Ib.vvvo - Ib_ref.vvvo
        s = numpy.linalg.norm(test) < self.thresh
        err = "error in Ib vvvo integrals"
        self.assertTrue(s,err)

        test = Ib.vovv - Ib_ref.vovv
        s = numpy.linalg.norm(test) < self.thresh
        err = "error in Ib vovv integrals"
        self.assertTrue(s,err)

        test = Ib.vvoo - Ib_ref.vvoo
        s = numpy.linalg.norm(test) < self.thresh
        err = "error in Ib vvoo integrals"
        self.assertTrue(s,err)

        test = Ib.vovo - Ib_ref.vovo
        s = numpy.linalg.norm(test) < self.thresh
        err = "error in Ib vovo integrals"
        self.assertTrue(s,err)

        test = Ib.oovv - Ib_ref.oovv
        s = numpy.linalg.norm(test) < self.thresh
        err = "error in Ib oovv integrals"
        self.assertTrue(s,err)

        test = Ib.vooo - Ib_ref.vooo
        s = numpy.linalg.norm(test) < self.thresh
        err = "error in Ib vooo integrals"
        self.assertTrue(s,err)

        test = Ib.ooov - Ib_ref.ooov
        s = numpy.linalg.norm(test) < self.thresh
        err = "error in Ib ooov integrals"
        self.assertTrue(s,err)

        test = Ib.oooo - Ib_ref.oooo
        s = numpy.linalg.norm(test) < self.thresh
        err = "error in Ib oooo integrals"
        self.assertTrue(s,err)

        test = Iabab.vvvv - Iabab_ref.vvvv
        s = numpy.linalg.norm(test) < self.thresh
        err = "error in Iab vvvv integrals"
        self.assertTrue(s,err)

        test = Iabab.vvvo - Iabab_ref.vvvo
        s = numpy.linalg.norm(test) < self.thresh
        err = "error in Iab vvvo integrals"
        self.assertTrue(s,err)

        test = Iabab.vvov - Iabab_ref.vvov
        s = numpy.linalg.norm(test) < self.thresh
        err = "error in Iab vvov integrals"
        self.assertTrue(s,err)

        test = Iabab.vovv - Iabab_ref.vovv
        s = numpy.linalg.norm(test) < self.thresh
        err = "error in Iab vovv integrals"
        self.assertTrue(s,err)

        test = Iabab.ovvv - Iabab_ref.ovvv
        s = numpy.linalg.norm(test) < self.thresh
        err = "error in Iab ovvv integrals"
        self.assertTrue(s,err)

        test = Iabab.vvoo - Iabab_ref.vvoo
        s = numpy.linalg.norm(test) < self.thresh
        err = "error in Iab vvoo integrals"
        self.assertTrue(s,err)

        test = Iabab.vovo - Iabab_ref.vovo
        s = numpy.linalg.norm(test) < self.thresh
        err = "error in Iab vovo integrals"
        self.assertTrue(s,err)

        test = Iabab.voov - Iabab_ref.voov
        s = numpy.linalg.norm(test) < self.thresh
        err = "error in Iab voov integrals"
        self.assertTrue(s,err)

        test = Iabab.ovov - Iabab_ref.ovov
        s = numpy.linalg.norm(test) < self.thresh
        err = "error in Iab ovov integrals"
        self.assertTrue(s,err)

        test = Iabab.ovvo - Iabab_ref.ovvo
        s = numpy.linalg.norm(test) < self.thresh
        err = "error in Iab ovvo integrals"
        self.assertTrue(s,err)

        test = Iabab.oovv - Iabab_ref.oovv
        s = numpy.linalg.norm(test) < self.thresh
        err = "error in Iab oovv integrals"
        self.assertTrue(s,err)

        test = Iabab.vooo - Iabab_ref.vooo
        s = numpy.linalg.norm(test) < self.thresh
        err = "error in Iab vooo integrals"
        self.assertTrue(s,err)

        test = Iabab.ovoo - Iabab_ref.ovoo
        s = numpy.linalg.norm(test) < self.thresh
        err = "error in Iab ovoo integrals"
        self.assertTrue(s,err)

        test = Iabab.ooov - Iabab_ref.ooov
        s = numpy.linalg.norm(test) < self.thresh
        err = "error in Iab ooov integrals"
        self.assertTrue(s,err)

        test = Iabab.oovo - Iabab_ref.oovo
        s = numpy.linalg.norm(test) < self.thresh
        err = "error in Iab oovo integrals"
        self.assertTrue(s,err)

        test = Iabab.oooo - Iabab_ref.oooo
        s = numpy.linalg.norm(test) < self.thresh
        err = "error in Iab oooo integrals"
        self.assertTrue(s,err)

    def test_Be_plus(self):
        from pyscf import gto, scf
        from cqcpy import integrals
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G',
            spin=1, charge=1)
        mf = scf.UHF(mol)
        mf.conv_tol = 1e-13
        Escf = mf.scf()
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        mo_occa = mo_occ[0]
        mo_occb = mo_occ[1]
        moa = mf.mo_coeff[0]
        mob = mf.mo_coeff[1]
        oa = (mf.mo_coeff[0])[:,mo_occa>0]
        va = (mf.mo_coeff[0])[:,mo_occa==0]
        ob = (mf.mo_coeff[1])[:,mo_occb>0]
        vb = (mf.mo_coeff[1])[:,mo_occb==0]
        noa = oa.shape[1]
        nva = va.shape[1]
        nob = ob.shape[1]
        nvb = vb.shape[1]
        Iaaaa = integrals.get_phys(mol,moa,moa,moa,moa)
        Iaaaa = test_utils.make_two_e_blocks_full(Iaaaa,noa,nva,noa,nva,noa,nva,noa,nva)
        Ibbbb = integrals.get_phys(mol,mob,mob,mob,mob)
        Ibbbb = test_utils.make_two_e_blocks_full(Ibbbb,nob,nvb,nob,nvb,nob,nvb,nob,nvb)
        Iabab = integrals.get_phys(mol,moa,mob,moa,mob)
        Iabab = test_utils.make_two_e_blocks_full(Iabab,noa,nva,nob,nvb,noa,nva,nob,nvb)
        I = spin_utils.int_to_spin(Iaaaa, Ibbbb, Iabab, noa, nva, nob, nvb)
        I_ref = integrals.eri_blocks(mf)
        z = I.vvvv - I_ref.vvvv
        s = numpy.linalg.norm(z) < self.thresh
        err = "error in vvvv block"
        self.assertTrue(s,err)

        z = I.vvvo - I_ref.vvvo
        s = numpy.linalg.norm(z) < self.thresh
        err = "error in vvvo block"
        self.assertTrue(s,err)

        z = I.vovv - I_ref.vovv
        s = numpy.linalg.norm(z) < self.thresh
        err = "error in vovo block"
        self.assertTrue(s,err)

        z = I.vovo - I_ref.vovo
        s = numpy.linalg.norm(z) < self.thresh
        err = "error in vovo block"
        self.assertTrue(s,err)

        z = I.vvoo - I_ref.vvoo
        s = numpy.linalg.norm(z) < self.thresh
        err = "error in vvoo block"
        self.assertTrue(s,err)

        z = I.oovv - I_ref.oovv
        s = numpy.linalg.norm(z) < self.thresh
        err = "error in oovv block"
        self.assertTrue(s,err)

        z = I.vooo - I_ref.vooo
        s = numpy.linalg.norm(z) < self.thresh
        err = "error in vooo block"
        self.assertTrue(s,err)

        z = I.ooov - I_ref.ooov
        s = numpy.linalg.norm(z) < self.thresh
        err = "error in ooov block"
        self.assertTrue(s,err)

        z = I.oooo - I_ref.oooo
        s = numpy.linalg.norm(z) < self.thresh
        err = "error in oooo block"
        self.assertTrue(s,err)

    def test_T(self):
        noa = 3
        nob = 2
        nva = 2
        nvb = 3
        no = noa + nob
        nv = nva + nvb
        T1ref,T2ref = test_utils.make_random_T(no,nv)
        Taa,Tab,Tbb = spin_utils.T2_to_spatial(T2ref, noa, nva, nob, nvb)
        Ta,Tb = spin_utils.T1_to_spatial(T1ref,noa,nva,nob,nvb)
        T2out = spin_utils.T2_to_spin(Taa,Tab,Tbb,noa,nva,nob,nvb)
        T1out = spin_utils.T1_to_spin(Ta,Tb,noa,nva,nob,nvb)
        z = T2out - T2ref
        s = numpy.linalg.norm(z < self.thresh)
        err = "Error in T2"
        self.assertTrue(s,err)
        z = T1out - T1ref
        s = numpy.linalg.norm(z < self.thresh)
        err = "Error in T1"
        self.assertTrue(s,err)

if __name__ == '__main__':
    unittest.main()

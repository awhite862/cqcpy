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
        T1, T2 = test_utils.make_random_T(no, nv)
        L1, L2 = test_utils.make_random_L(no, nv)

        pba_ref = cc_equations.ccsd_1rdm_ba(T1, T2, L1, L2)
        pba_out = cc_equations.ccsd_1rdm_ba_opt(T1, T2, L1, L2)
        diff = numpy.linalg.norm(pba_ref - pba_out)/numpy.sqrt(pba_ref.size)
        self.assertTrue(diff < thresh, "Error in p_ba: {}".format(diff))

        pji_ref = cc_equations.ccsd_1rdm_ji(T1, T2, L1, L2)
        pji_out = cc_equations.ccsd_1rdm_ji_opt(T1, T2, L1, L2)
        diff = numpy.linalg.norm(pji_ref - pji_out)/numpy.sqrt(pji_ref.size)
        self.assertTrue(diff < thresh, "Error in p_ji: {}".format(diff))

        pai_ref = cc_equations.ccsd_1rdm_ai(T1, T2, L1, L2)
        pai_out = cc_equations.ccsd_1rdm_ai_opt(T1, T2, L1, L2)
        diff = numpy.linalg.norm(pai_ref - pai_out)/numpy.sqrt(pai_ref.size)
        self.assertTrue(diff < thresh, "Error in p_ai: {}".format(diff))

    def test_2rdm_opt(self):
        no = 4
        nv = 8
        thresh = 1e-12
        T1, T2 = test_utils.make_random_T(no, nv)
        L1, L2 = test_utils.make_random_L(no, nv)

        pcdab_ref = cc_equations.ccsd_2rdm_cdab(T1, T2, L1, L2)
        pcdab_out = cc_equations.ccsd_2rdm_cdab_opt(T1, T2, L1, L2)
        diff = numpy.linalg.norm(pcdab_ref - pcdab_out)
        diff /= numpy.sqrt(pcdab_ref.size)
        self.assertTrue(diff < thresh, "Error in p_cdab: {}".format(diff))

        pbcai_ref = cc_equations.ccsd_2rdm_bcai(T1, T2, L1, L2)
        pbcai_out = cc_equations.ccsd_2rdm_bcai_opt(T1, T2, L1, L2)
        diff = numpy.linalg.norm(pbcai_ref - pbcai_out)
        diff /= numpy.sqrt(pbcai_ref.size)
        self.assertTrue(diff < thresh, "Error in p_bcai: {}".format(diff))

        pbjai_ref = cc_equations.ccsd_2rdm_bjai(T1, T2, L1, L2)
        pbjai_out = cc_equations.ccsd_2rdm_bjai_opt(T1, T2, L1, L2)
        diff = numpy.linalg.norm(pbjai_ref - pbjai_out)
        diff /= numpy.sqrt(pbjai_ref.size)
        self.assertTrue(diff < thresh, "Error in p_bjai: {}".format(diff))

        pabij_ref = cc_equations.ccsd_2rdm_abij(T1, T2, L1, L2)
        pabij_out = cc_equations.ccsd_2rdm_abij_opt(T1, T2, L1, L2)
        diff = numpy.linalg.norm(pabij_ref - pabij_out)
        diff /= numpy.sqrt(pabij_ref.size)
        self.assertTrue(diff < thresh, "Error in p_abij: {}".format(diff))

        pkaij_ref = cc_equations.ccsd_2rdm_kaij(T1, T2, L1, L2)
        pkaij_out = cc_equations.ccsd_2rdm_kaij_opt(T1, T2, L1, L2)
        diff = numpy.linalg.norm(pkaij_ref - pkaij_out)
        diff /= numpy.sqrt(pkaij_ref.size)
        self.assertTrue(diff < thresh, "Error in p_kaij: {}".format(diff))

        pklij_ref = cc_equations.ccsd_2rdm_klij(T1, T2, L1, L2)
        pklij_out = cc_equations.ccsd_2rdm_klij_opt(T1, T2, L1, L2)
        diff = numpy.linalg.norm(pklij_ref - pklij_out)
        diff /= numpy.sqrt(pklij_ref.size)
        self.assertTrue(diff < thresh, "Error in p_klij: {}".format(diff))

    def test_u1rdm(self):
        noa = 3
        nva = 5
        nob = 2
        nvb = 6
        thresh = 1e-14

        # use unrestricted one-particle property
        Aa = test_utils.make_random_F(noa, nva)
        Ab = test_utils.make_random_F(nob, nvb)
        Atot = spin_utils.F_to_spin(Aa, Ab, noa, nva, nob, nvb)

        # get unrestricted and general amplitudes
        T1a, T1b = test_utils.make_random_T1_spatial(noa, nva, nob, nvb)
        T2aa, T2ab, T2bb \
            = test_utils.make_random_T2_spatial(noa, nva, nob, nvb)
        L1a, L1b = test_utils.make_random_T1_spatial(nva, noa, nvb, nob)
        L2aa, L2ab, L2bb \
            = test_utils.make_random_T2_spatial(nva, noa, nvb, nob)
        T1 = spin_utils.T1_to_spin(T1a, T1b, noa, nva, nob, nvb)
        L1 = spin_utils.T1_to_spin(L1a, L1b, nva, noa, nvb, nob)
        T2 = spin_utils.T2_to_spin(T2aa, T2ab, T2bb, noa, nva, nob, nvb)
        L2 = spin_utils.T2_to_spin(L2aa, L2ab, L2bb, nva, noa, nvb, nob)

        # make general pieces of 1-rdm
        pia = L1.copy()
        pba = cc_equations.ccsd_1rdm_ba_opt(T1, T2, L1, L2)
        pji = cc_equations.ccsd_1rdm_ji_opt(T1, T2, L1, L2)
        pai = cc_equations.ccsd_1rdm_ai_opt(T1, T2, L1, L2)

        # make unrestricted 1-rdm
        pia_a = L1a.copy()
        pia_b = L1b.copy()
        pba_a, pba_b = cc_equations.uccsd_1rdm_ba(
            T1a, T1b, T2aa, T2ab, T2bb, L1a, L1b, L2aa, L2ab, L2bb)
        pji_a, pji_b = cc_equations.uccsd_1rdm_ji(
            T1a, T1b, T2aa, T2ab, T2bb, L1a, L1b, L2aa, L2ab, L2bb)
        pai_a, pai_b = cc_equations.uccsd_1rdm_ai(
            T1a, T1b, T2aa, T2ab, T2bb, L1a, L1b, L2aa, L2ab, L2bb)

        # ia
        ref = numpy.einsum('ia,ai->', pia, Atot.vo)
        out = numpy.einsum('ia,ai->', pia_a, Aa.vo)
        out += numpy.einsum('ia,ai->', pia_b, Ab.vo)
        diff = abs(out - ref) / abs(ref)
        self.assertTrue(diff < thresh, "Error in Pia: {}".format(diff))

        # ba
        ref = numpy.einsum('ba,ab->', pba, Atot.vv)
        out = numpy.einsum('ba,ab->', pba_a, Aa.vv)
        out += numpy.einsum('ba,ab->', pba_b, Ab.vv)
        diff = abs(out - ref) / abs(ref)
        self.assertTrue(diff < thresh, "Error in Pba: {}".format(diff))

        # ji
        ref = numpy.einsum('ji,ij->', pji, Atot.oo)
        out = numpy.einsum('ji,ij->', pji_a, Aa.oo)
        out += numpy.einsum('ji,ij->', pji_b, Ab.oo)
        diff = abs(out - ref) / abs(ref)
        self.assertTrue(diff < thresh, "Error in Pji: {}".format(diff))

        # ai
        ref = numpy.einsum('ai,ia->', pai, Atot.ov)
        out = numpy.einsum('ai,ia->', pai_a, Aa.ov)
        out += numpy.einsum('ai,ia->', pai_b, Ab.ov)
        diff = abs(out - ref) / abs(ref)
        self.assertTrue(diff < thresh, "Error in Pai: {}".format(diff))

    def test_u2rdm(self):
        noa = 3
        nva = 5
        nob = 2
        nvb = 6
        thresh = 1e-14

        # use unrestricted one-particle property
        Aa = test_utils.make_random_I_anti(noa, nva)
        Ab = test_utils.make_random_I_anti(nob, nvb)
        Aab = test_utils.make_random_Ifull_gen(
            noa, nva, nob, nvb, noa, nva, nob, nvb)
        Atot = spin_utils.int_to_spin2(Aa, Ab, Aab, noa, nva, nob, nvb)

        # get unrestricted and general amplitudes
        T1a, T1b = test_utils.make_random_T1_spatial(noa, nva, nob, nvb)
        T2aa, T2ab, T2bb \
            = test_utils.make_random_T2_spatial(noa, nva, nob, nvb)
        L1a, L1b = test_utils.make_random_T1_spatial(nva, noa, nvb, nob)
        L2aa, L2ab, L2bb \
            = test_utils.make_random_T2_spatial(nva, noa, nvb, nob)
        T1 = spin_utils.T1_to_spin(T1a, T1b, noa, nva, nob, nvb)
        L1 = spin_utils.T1_to_spin(L1a, L1b, nva, noa, nvb, nob)
        T2 = spin_utils.T2_to_spin(T2aa, T2ab, T2bb, noa, nva, nob, nvb)
        L2 = spin_utils.T2_to_spin(L2aa, L2ab, L2bb, nva, noa, nvb, nob)

        # make general pieces of 2-rdm
        Pijab = L2.copy()
        Pciab = cc_equations.ccsd_2rdm_ciab(T1, T2, L1, L2)
        Pjkai = cc_equations.ccsd_2rdm_jkai(T1, T2, L1, L2)
        Pcdab = cc_equations.ccsd_2rdm_cdab(T1, T2, L1, L2)
        Pbjai = cc_equations.ccsd_2rdm_bjai(T1, T2, L1, L2)
        Pklij = cc_equations.ccsd_2rdm_klij(T1, T2, L1, L2)
        Pbcai = cc_equations.ccsd_2rdm_bcai(T1, T2, L1, L2)
        Pkaij = cc_equations.ccsd_2rdm_kaij(T1, T2, L1, L2)
        Pabij = cc_equations.ccsd_2rdm_abij(T1, T2, L1, L2)

        # make unrestricted RDMs
        Pijab_u = L2aa.copy()
        PIJAB_u = L2bb.copy()
        PiJaB_u = L2ab.copy()

        Pciab_u, PCIAB_u, PcIaB_u, PCiAb_u = cc_equations.uccsd_2rdm_ciab(
            T1a, T1b, T2aa, T2ab, T2bb, L1a, L1b, L2aa, L2ab, L2bb)
        Pjkai_u, PJKAI_u, PjKaI_u, PJkAi_u = cc_equations.uccsd_2rdm_jkai(
            T1a, T1b, T2aa, T2ab, T2bb, L1a, L1b, L2aa, L2ab, L2bb)
        Pcdab_u, PCDAB_u, PcDaB_u = cc_equations.uccsd_2rdm_cdab(
            T1a, T1b, T2aa, T2ab, T2bb, L1a, L1b, L2aa, L2ab, L2bb)
        Pbjai_u, PBJAI_u, PbJaI_u, PbJAi_u, PBjaI_u, PBjAi_u \
            = cc_equations.uccsd_2rdm_bjai(
                T1a, T1b, T2aa, T2ab, T2bb, L1a, L1b, L2aa, L2ab, L2bb)
        Pklij_u, PKLIJ_u, PkLiJ_u = cc_equations.uccsd_2rdm_klij(
            T1a, T1b, T2aa, T2ab, T2bb, L1a, L1b, L2aa, L2ab, L2bb)
        Pbcai_u, PBCAI_u, PbCaI_u, PBcAi_u = cc_equations.uccsd_2rdm_bcai(
            T1a, T1b, T2aa, T2ab, T2bb, L1a, L1b, L2aa, L2ab, L2bb)
        Pkaij_u, PKAIJ_u, PkAiJ_u, PKaIj_u = cc_equations.uccsd_2rdm_kaij(
            T1a, T1b, T2aa, T2ab, T2bb, L1a, L1b, L2aa, L2ab, L2bb)
        Pabij_u, PABIJ_u, PaBiJ_u = cc_equations.uccsd_2rdm_abij(
            T1a, T1b, T2aa, T2ab, T2bb, L1a, L1b, L2aa, L2ab, L2bb)

        # ijab
        ref = numpy.einsum('ijab,abij->', Pijab, Atot.vvoo)
        out = numpy.einsum('ijab,abij->', Pijab_u, Aa.vvoo)
        out += numpy.einsum('ijab,abij->', PIJAB_u, Ab.vvoo)
        out += 4.0*numpy.einsum('ijab,abij->', PiJaB_u, Aab.vvoo)
        diff = abs(out - ref) / abs(ref + 0.001)
        self.assertTrue(diff < thresh, "Error in Pijab: {}".format(diff))

        # ciab
        ref = numpy.einsum('ciab,abci->', Pciab, Atot.vvvo)
        out = numpy.einsum('ciab,abci->', Pciab_u, Aa.vvvo)
        out += numpy.einsum('ciab,abci->', PCIAB_u, Ab.vvvo)
        out += 2.0*numpy.einsum('ciab,abci->', PcIaB_u, Aab.vvvo)
        out += 2.0*numpy.einsum('ciab,baic->', PCiAb_u, Aab.vvov)
        diff = abs(out - ref) / abs(ref + 0.001)
        self.assertTrue(diff < thresh, "Error in Pciab: {}".format(diff))

        # jkai
        ref = numpy.einsum('jkai,aijk->', Pjkai, Atot.vooo)
        out = numpy.einsum('jkai,aijk->', Pjkai_u, Aa.vooo)
        out += numpy.einsum('jkai,aijk->', PJKAI_u, Ab.vooo)
        out += 2.0*numpy.einsum('jKaI,aIjK->', PjKaI_u, Aab.vooo)
        out += 2.0*numpy.einsum('JkAi,iAkJ->', PJkAi_u, Aab.ovoo)
        diff = abs(out - ref) / abs(ref + 0.001)
        self.assertTrue(diff < thresh, "Error in Pciab: {}".format(diff))

        # cdab
        ref = numpy.einsum('cdab,abcd->', Pcdab, Atot.vvvv)
        out = numpy.einsum('cdab,abcd->', Pcdab_u, Aa.vvvv)
        out += numpy.einsum('cdab,abcd->', PCDAB_u, Ab.vvvv)
        out += 4.0*numpy.einsum('cdab,abcd->', PcDaB_u, Aab.vvvv)
        diff = abs(out - ref) / abs(ref + 0.001)
        self.assertTrue(diff < thresh, "Error in Pcdab: {}".format(diff))

        # bjai
        ref = numpy.einsum('bjai,aibj->', Pbjai, Atot.vovo)
        out = numpy.einsum('bjai,aibj->', Pbjai_u, Aa.vovo)
        out += numpy.einsum('bJaI,aIbJ->', PbJaI_u, Aab.vovo)
        out -= numpy.einsum('bJAi,iAbJ->', PbJAi_u, Aab.ovvo)
        out -= numpy.einsum('BjaI,aIjB->', PBjaI_u, Aab.voov)
        out += numpy.einsum('BjAi,iAjB->', PBjAi_u, Aab.ovov)
        out += numpy.einsum('BJAI,AIBJ->', PBJAI_u, Ab.vovo)
        diff = abs(out - ref) / abs(ref + 0.001)
        self.assertTrue(diff < thresh, "Error in Pbjai: {}".format(diff))

        # klij
        ref = numpy.einsum('klij,ijkl->', Pklij, Atot.oooo)
        out = numpy.einsum('klij,ijkl->', Pklij_u, Aa.oooo)
        out += numpy.einsum('klij,ijkl->', PKLIJ_u, Ab.oooo)
        out += 4.0*numpy.einsum('kLiJ,iJkL->', PkLiJ_u, Aab.oooo)
        diff = abs(out - ref) / abs(ref + 0.001)
        self.assertTrue(diff < thresh, "Error in Pklij: {}".format(diff))

        # bcai
        ref = numpy.einsum('bcai,aibc->', Pbcai, Atot.vovv)
        out = numpy.einsum('bcai,aibc->', Pbcai_u, Aa.vovv)
        out += 2.0*numpy.einsum('bCaI,aIbC->', PbCaI_u, Aab.vovv)
        out += 2.0*numpy.einsum('BcAi,iAcB->', PBcAi_u, Aab.ovvv)
        out += numpy.einsum('bcai,aibc->', PBCAI_u, Ab.vovv)
        diff = abs(out - ref) / abs(ref + 0.001)
        self.assertTrue(diff < thresh, "Error in Pbcai: {}".format(diff))

        # kaij
        ref = numpy.einsum('kaij,ijka->', Pkaij, Atot.ooov)
        out = numpy.einsum('kaij,ijka->', Pkaij_u, Aa.ooov)
        out += 2.0*numpy.einsum('kaij,ijka->', PkAiJ_u, Aab.ooov)
        out += 2.0*numpy.einsum('KaIj,jIaK->', PKaIj_u, Aab.oovo)
        out += numpy.einsum('kaij,ijka->', PKAIJ_u, Ab.ooov)
        diff = abs(out - ref) / abs(ref + 0.001)
        self.assertTrue(diff < thresh, "Error in Pkaij: {}".format(diff))

        # abij
        ref = numpy.einsum('abij,ijab->', Pabij, Atot.oovv)
        out = numpy.einsum('abij,ijab->', Pabij_u, Aa.oovv)
        out += numpy.einsum('abij,ijab->', PABIJ_u, Ab.oovv)
        out += 4.0*numpy.einsum('aBiJ,iJaB->', PaBiJ_u, Aab.oovv)
        diff = abs(out - ref) / abs(ref + 0.001)
        self.assertTrue(diff < thresh, "Error in Pabij: {}".format(diff))

    def test_r1rdm(self):
        no = 3
        nv = 5
        thresh = 1e-12

        T1 = numpy.random.random((nv, no))
        T2 = numpy.random.random((nv, nv, no, no))
        T2 = T2 + T2.transpose((1, 0, 3, 2))

        L1 = numpy.random.random((no, nv))
        L2 = numpy.random.random((no, no, nv, nv))
        L2 = L2 + L2.transpose((1, 0, 3, 2))

        T1a = T1b = T1
        T2aa = T2 - T2.transpose((0, 1, 3, 2))

        L1a = L1b = L1
        L2aa = L2 - L2.transpose((0, 1, 3, 2))

        # make unrestricted 1-rdm
        pia_a = L1a.copy()
        pba_a, pba_b = cc_equations.uccsd_1rdm_ba(
            T1a, T1b, T2aa, T2, T2aa, L1a, L1b, L2aa, L2, L2aa)
        pji_a, pji_b = cc_equations.uccsd_1rdm_ji(
            T1a, T1b, T2aa, T2, T2aa, L1a, L1b, L2aa, L2, L2aa)
        pai_a, pai_b = cc_equations.uccsd_1rdm_ai(
            T1a, T1b, T2aa, T2, T2aa, L1a, L1b, L2aa, L2, L2aa)

        pia = L1.copy()
        pba = cc_equations.rccsd_1rdm_ba(T1, T2, L1, L2)
        pji = cc_equations.rccsd_1rdm_ji(T1, T2, L1, L2)
        pai = cc_equations.rccsd_1rdm_ai(T1, T2, L1, L2)

        diff = numpy.linalg.norm(pia_a - pia)/numpy.sqrt(pia_a.size)
        self.assertTrue(diff < thresh, "Error in p_ia: {}".format(diff))

        diff = numpy.linalg.norm(pba_a - pba)/numpy.sqrt(pba_a.size)
        self.assertTrue(diff < thresh, "Error in p_ab: {}".format(diff))

        diff = numpy.linalg.norm(pji_a - pji)/numpy.sqrt(pji_a.size)
        self.assertTrue(diff < thresh, "Error in p_ji: {}".format(diff))

        diff = numpy.linalg.norm(pai_a - pai)/numpy.sqrt(pai_a.size)
        self.assertTrue(diff < thresh, "Error in p_ai: {}".format(diff))

    def test_r2rdm(self):
        no = 3
        nv = 5
        thresh = 1e-12

        T1 = numpy.random.random((nv, no))
        T2 = numpy.random.random((nv, nv, no, no))
        T2 = T2 + T2.transpose((1, 0, 3, 2))

        L1 = numpy.random.random((no, nv))
        L2 = numpy.random.random((no, no, nv, nv))
        L2 = L2 + L2.transpose((1, 0, 3, 2))

        T1a = T1b = T1
        T2aa = T2 - T2.transpose((0, 1, 3, 2))

        L1a = L1b = L1
        L2aa = L2 - L2.transpose((0, 1, 3, 2))

        # make unrestricted 2-rdm
        PiJaB_u = L2.copy()

        Pciab_u, PCIAB_u, PcIaB_u, PCiAb_u = cc_equations.uccsd_2rdm_ciab(
            T1a, T1b, T2aa, T2, T2aa, L1a, L1b, L2aa, L2, L2aa)
        Pjkai_u, PJKAI_u, PjKaI_u, PJkAi_u = cc_equations.uccsd_2rdm_jkai(
            T1a, T1b, T2aa, T2, T2aa, L1a, L1b, L2aa, L2, L2aa)
        Pcdab_u, PCDAB_u, PcDaB_u = cc_equations.uccsd_2rdm_cdab(
            T1a, T1b, T2aa, T2, T2aa, L1a, L1b, L2aa, L2, L2aa)
        Pbjai_u, PBJAI_u, PbJaI_u, PbJAi_u, PBjaI_u, PBjAi_u \
            = cc_equations.uccsd_2rdm_bjai(
                T1a, T1b, T2aa, T2, T2aa, L1a, L1b, L2aa, L2, L2aa)
        Pklij_u, PKLIJ_u, PkLiJ_u = cc_equations.uccsd_2rdm_klij(
            T1a, T1b, T2aa, T2, T2aa, L1a, L1b, L2aa, L2, L2aa)
        Pbcai_u, PBCAI_u, PbCaI_u, PBcAi_u = cc_equations.uccsd_2rdm_bcai(
            T1a, T1b, T2aa, T2, T2aa, L1a, L1b, L2aa, L2, L2aa)
        Pkaij_u, PKAIJ_u, PkAiJ_u, PKaIj_u = cc_equations.uccsd_2rdm_kaij(
            T1a, T1b, T2aa, T2, T2aa, L1a, L1b, L2aa, L2, L2aa)
        Pabij_u, PABIJ_u, PaBiJ_u = cc_equations.uccsd_2rdm_abij(
            T1a, T1b, T2aa, T2, T2aa, L1a, L1b, L2aa, L2, L2aa)

        # make restricted 2-rdm
        Pijab = L2.copy()

        Pciab = cc_equations.rccsd_2rdm_ciab(T1, T2, L1, L2)
        Pjkai = cc_equations.rccsd_2rdm_jkai(T1, T2, L1, L2)
        Pcdab = cc_equations.rccsd_2rdm_cdab(T1, T2, L1, L2)
        Pbjai = cc_equations.rccsd_2rdm_bjai(T1, T2, L1, L2)
        Pbjia = cc_equations.rccsd_2rdm_bjia(T1, T2, L1, L2)
        Pklij = cc_equations.rccsd_2rdm_klij(T1, T2, L1, L2)
        Pbcai = cc_equations.rccsd_2rdm_bcai(T1, T2, L1, L2)
        Pkaij = cc_equations.rccsd_2rdm_kaij(T1, T2, L1, L2)
        Pabij = cc_equations.rccsd_2rdm_abij(T1, T2, L1, L2)

        # ijab
        diff = numpy.linalg.norm(Pijab - PiJaB_u)/numpy.sqrt(PiJaB_u.size)
        self.assertTrue(diff < thresh, "Error in Pijab: {}".format(diff))

        # ciab
        diff = numpy.linalg.norm(Pciab - PcIaB_u)/numpy.sqrt(PcIaB_u.size)
        self.assertTrue(diff < thresh, "Error in Picab: {}".format(diff))

        # jkai
        diff = numpy.linalg.norm(Pjkai - PjKaI_u)/numpy.sqrt(PjKaI_u.size)
        self.assertTrue(diff < thresh, "Error in Pjkai: {}".format(diff))

        # cdab
        diff = numpy.linalg.norm(Pcdab - PcDaB_u)/numpy.sqrt(PcDaB_u.size)
        self.assertTrue(diff < thresh, "Error in Pcdab: {}".format(diff))

        # bjai
        diff = numpy.linalg.norm(Pbjai - PbJaI_u)/numpy.sqrt(PbJaI_u.size)
        self.assertTrue(diff < thresh, "Error in Pbjai: {}".format(diff))

        # bjia
        diff = numpy.linalg.norm(Pbjia + PbJAi_u.transpose((0, 1, 3, 2)))
        diff /= numpy.sqrt(PbJAi_u.size)
        self.assertTrue(diff < thresh, "Error in Pbjai: {}".format(diff))

        # klij
        diff = numpy.linalg.norm(Pklij - PkLiJ_u)/numpy.sqrt(PkLiJ_u.size)
        self.assertTrue(diff < thresh, "Error in Pklij: {}".format(diff))

        # bcai
        diff = numpy.linalg.norm(Pbcai - PbCaI_u)/numpy.sqrt(PbCaI_u.size)
        self.assertTrue(diff < thresh, "Error in Pbcai: {}".format(diff))

        # kaij
        diff = numpy.linalg.norm(Pkaij - PkAiJ_u)/numpy.sqrt(PkAiJ_u.size)
        self.assertTrue(diff < thresh, "Error in Pkaij: {}".format(diff))

        # abij
        diff = numpy.linalg.norm(Pabij - PaBiJ_u)/numpy.sqrt(PaBiJ_u.size)
        self.assertTrue(diff < thresh, "Error in Pabij: {}".format(diff))


if __name__ == '__main__':
    unittest.main()

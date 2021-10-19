import unittest
import numpy


class IntegralsTest(unittest.TestCase):
    def _get_mf(self):
        from pyscf import gto, scf
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G',
            spin=1, charge=1)
        mf = scf.UHF(mol)
        mf.conv_tol = 1e-13
        mf.scf()
        return mf

    def _get_pbc(self):
        import pyscf.pbc.gto as pbc_gto
        import pyscf.pbc.scf as pbc_scf
        cell = pbc_gto.Cell()
        cell.atom = '''
        C 0.000000000000   0.000000000000   0.000000000000
        C 1.685068664391   1.685068664391   1.685068664391
        '''
        cell.basis = 'gth-szv'
        cell.pseudo = 'gth-pade'
        cell.a = '''
        0.000000000, 3.370137329, 3.370137329
        3.370137329, 0.000000000, 3.370137329
        3.370137329, 3.370137329, 0.000000000'''
        cell.unit = 'B'
        cell.verbose = 0
        cell.build()
        pbc_mf = pbc_scf.RHF(cell, exxdiv=None)
        pbc_mf.kernel()
        return pbc_mf

    def _get_pbc2(self):
        import pyscf.pbc.gto as pbc_gto
        import pyscf.pbc.scf as pbc_scf
        cell = pbc_gto.Cell()
        cell.atom = '''
        C 0.000000000000   0.000000000000   0.000000000000
        C 1.685068664391   1.685068664391   1.685068664391
        '''
        cell.basis = 'gth-szv'
        cell.pseudo = 'gth-pade'
        cell.a = '''
        0.000000000, 3.370137329, 3.370137329
        3.370137329, 0.000000000, 3.370137329
        3.370137329, 3.370137329, 0.000000000'''
        cell.unit = 'B'
        cell.verbose = 0
        cell.build()
        kpt = cell.make_kpts((1, 1, 1), scaled_center=(0, 0, 1./3.))
        pbc_mf2 = pbc_scf.RHF(cell, kpt=kpt, exxdiv=None)
        pbc_mf2.kernel()
        return pbc_mf2

    def _get_pbc3(self):
        import pyscf.pbc.gto as pbc_gto
        import pyscf.pbc.scf as pbc_scf
        cell = pbc_gto.Cell()
        cell.atom = '''
        C 0.000000000000   0.000000000000   0.000000000000
        C 1.685068664391   1.685068664391   1.685068664391
        '''
        cell.basis = 'gth-szv'
        cell.pseudo = 'gth-pade'
        cell.a = '''
        0.000000000, 3.370137329, 3.370137329
        3.370137329, 0.000000000, 3.370137329
        3.370137329, 3.370137329, 0.000000000'''
        cell.unit = 'B'
        cell.verbose = 0
        cell.build()
        kpts = cell.make_kpts((1, 1, 2))
        kmf = pbc_scf.KRHF(cell, kpts, exxdiv=None)
        kmf.kernel()
        return kmf

    def test_phys(self):
        from cqcpy import integrals
        mf = self._get_mf()
        nao = mf.mo_coeff[0].shape[0]
        mo1 = mf.mo_coeff[0][:, 0].reshape((nao, 1))
        mo2 = mf.mo_coeff[0][:, 1].reshape((nao, 1))
        mo3 = mf.mo_coeff[1][:, 0].reshape((nao, 1))
        mo4 = mf.mo_coeff[1][:, 1].reshape((nao, 1))

        ref = integrals.get_chem(
            mf.mol, mo1, mo3, mo2, mo4, anti=True).transpose((0, 2, 1, 3))
        out = integrals.get_phys(mf.mol, mo1, mo2, mo3, mo4, anti=True)
        diff = abs(ref[0, 0, 0, 0] - out[0, 0, 0, 0])
        self.assertTrue(diff < 1e-12)

    def test_u(self):
        from cqcpy import integrals
        mf = self._get_mf()
        moa = mf.mo_coeff[0]
        mob = mf.mo_coeff[1]
        Iref = integrals.get_chemu(
            mf.mol, moa, moa, moa, moa, mob, mob, mob, mob, anti=True)
        Iout = integrals.get_chemu_all(mf.mol, moa, mob, anti=True)
        diff = numpy.linalg.norm(Iref - Iout)
        self.assertTrue(diff < 1e-12)

    def test_sol_phys(self):
        from cqcpy import integrals
        pbc_mf = self._get_pbc()
        mo_coeff = pbc_mf.mo_coeff
        nao = mo_coeff.shape[0]
        mo1 = mo_coeff[:, 0].reshape((nao, 1))
        mo2 = mo_coeff[:, 1].reshape((nao, 1))
        mo3 = mo_coeff[:, 2].reshape((nao, 1))
        mo4 = mo_coeff[:, 3].reshape((nao, 1))

        ref = integrals.get_chem_sol(
            pbc_mf, mo1, mo3, mo2, mo4, anti=True).transpose((0, 2, 1, 3))
        out = integrals.get_phys_sol(
            pbc_mf, mo1, mo2, mo3, mo4, anti=True)
        diff = abs(ref[0, 0, 0, 0] - out[0, 0, 0, 0])
        self.assertTrue(diff < 1e-12)

    def test_fock(self):
        from cqcpy import integrals
        from cqcpy import utils
        mf = self._get_mf()
        f = mf.get_fock()
        h1 = mf.get_hcore(mf.mol)
        va = f[0] - h1
        vb = f[1] - h1
        moa = mf.mo_coeff[0]
        mob = mf.mo_coeff[1]
        vmoa = numpy.einsum('mp,mn,nq->pq', numpy.conj(moa), va, moa)
        vmob = numpy.einsum('mp,mn,nq->pq', numpy.conj(mob), vb, mob)

        na = mf.mol.nelectron // 2 + mf.mol.spin
        nb = mf.mol.nelectron - na
        nmo = moa.shape[1]
        pad = numpy.zeros(nmo)
        pbd = numpy.zeros(nmo)
        pad[:na] = numpy.ones(na)
        pbd[:nb] = numpy.ones(nb)
        pmoa = numpy.diag(pad)
        pmob = numpy.diag(pbd)

        p = utils.block_diag(pmoa, pmob)
        eri = integrals.get_physu_all(mf.mol, moa, mob, anti=True)
        JK = numpy.einsum('pqrs,qs->pr', eri, p)
        vtot = utils.block_diag(vmoa, vmob)
        diff = numpy.linalg.norm(vtot - JK)
        self.assertTrue(diff < 1e-12)

    def test_fock_sol(self):
        from cqcpy import integrals
        from cqcpy import utils
        mf = self._get_pbc()
        f = mf.get_fock()
        h1 = mf.get_hcore(mf.cell)
        v = f - h1
        mo = mf.mo_coeff
        vmo = numpy.einsum('mp,mn,nq->pq', numpy.conj(mo), v, mo)

        na = mf.mol.nelectron // 2 + mf.mol.spin
        nb = mf.mol.nelectron - na
        nmo = mo.shape[1]
        pad = numpy.zeros(nmo)
        pbd = numpy.zeros(nmo)
        pad[:na] = numpy.ones(na)
        pbd[:nb] = numpy.ones(nb)
        pmoa = numpy.diag(pad)
        pmob = numpy.diag(pbd)

        p = utils.block_diag(pmoa, pmob)
        eri = integrals.get_physu_all_sol(mf, mo, mo, anti=True)
        JK = numpy.einsum('pqrs,qs->pr', eri, p)
        vtot = utils.block_diag(vmo, vmo)
        diff = numpy.linalg.norm(vtot - JK)
        self.assertTrue(diff < 1e-12)

    def test_fock_sol_k(self):
        from cqcpy import integrals
        from cqcpy import utils
        mf = self._get_pbc2()
        f = mf.get_fock()
        h1 = mf.get_hcore(mf.cell)
        v = f - h1
        mo = mf.mo_coeff
        vmo = numpy.einsum('mp,mn,nq->pq', numpy.conj(mo), v, mo)

        na = mf.mol.nelectron // 2 + mf.mol.spin
        nb = mf.mol.nelectron - na
        nmo = mo.shape[1]
        pad = numpy.zeros(nmo)
        pbd = numpy.zeros(nmo)
        pad[:na] = numpy.ones(na)
        pbd[:nb] = numpy.ones(nb)
        pmoa = numpy.diag(pad)
        pmob = numpy.diag(pbd)

        p = utils.block_diag(pmoa, pmob)
        eri = integrals.get_physu_all_sol(mf, mo, mo, anti=True)
        JK = numpy.einsum('pqrs,sq->pr', eri, p)
        vtot = utils.block_diag(vmo, vmo)
        diff = numpy.linalg.norm(vtot - JK)
        self.assertTrue(diff < 1e-12)

    def test_eri_sol_k(self):
        from cqcpy import integrals
        import pyscf.pbc.cc as pbc_cc
        mf = self._get_pbc2()
        cc = pbc_cc.CCSD(mf)
        eris = cc.ao2mo()
        mo = mf.mo_coeff
        o = mo[:, mf.mo_occ > 0]
        v = mo[:, mf.mo_occ == 0]

        oooo = eris.oooo
        Ioooo = integrals.get_phys_gen(mf, o, o, o, o)
        diff = numpy.linalg.norm(oooo - Ioooo.transpose((0, 2, 1, 3)))
        self.assertTrue(diff < 1e-12)

        oovv = eris.oovv
        Ioovv = integrals.get_chem_sol(mf, o, o, v, v)
        diff = numpy.linalg.norm(oovv - Ioovv)
        self.assertTrue(diff < 1e-12)

    def test_eri_sol_kpts(self):
        from cqcpy import integrals
        import pyscf.pbc.cc as pbc_cc
        kmf = self._get_pbc3()
        cc = pbc_cc.KCCSD(kmf)
        eris = cc.ao2mo()
        mo = kmf.mo_coeff
        nocc = cc.nocc
        nks = len(mo)
        k1 = 0
        k2 = 1
        k3 = 0
        k4 = 1
        o1 = mo[k1][:, kmf.mo_occ[k1] > 0]
        o2 = mo[k2][:, kmf.mo_occ[k2] > 0]
        o3 = mo[k3][:, kmf.mo_occ[k3] > 0]
        o4 = mo[k4][:, kmf.mo_occ[k4] > 0]
        v3 = mo[k3][:, kmf.mo_occ[k3] == 0]
        v4 = mo[k4][:, kmf.mo_occ[k4] == 0]

        # test oooo in chemist's notation
        oooo = eris.oooo
        Ioooo = 1/nks*integrals.get_chem_solk(kmf, (k1, k3, k2, k4), o1, o3, o2, o4)
        diff = numpy.linalg.norm(oooo[k1, k2, k3] - Ioooo.transpose((0, 2, 1, 3)))
        self.assertTrue(diff < 1e-12)

        # test oooo in physicists notation
        Ioooo = 1/nks*integrals.get_phys_solk(kmf, (k1, k2, k3, k4), o1, o2, o3, o4)
        diff = numpy.linalg.norm(oooo[k1, k2, k3] - Ioooo)
        self.assertTrue(diff < 1e-12)

        # test oovv in chemists notation
        oovv = eris.oovv
        Iovov = 1/nks*integrals.get_chem_solk(kmf, (k1, k3, k2, k4), o1, v3, o2, v4)
        diff = numpy.linalg.norm(oovv[k1, k2, k3] - Iovov.transpose((0, 2, 1, 3)))
        self.assertTrue(diff < 1e-12)

    def test_integrals_kpts(self):
        from cqcpy import integrals
        kmf = self._get_pbc3()
        nocc = numpy.count_nonzero(kmf.mo_occ[0])
        Eref = kmf.energy_tot()
        nkpts = len(kmf.kpts)

        # test in chemists notation
        h1, h2 = integrals.get_solk_integrals(kmf, eriorder='chem')
        h2 = h2/nkpts
        Eout = kmf.energy_nuc()
        for k in range(nkpts):
            Eout += 2.0*numpy.einsum('ii->', h1[k, :nocc, :nocc]) / nkpts

        for kp in range(nkpts):
            for kq in range(nkpts):
                Eout += 2.0*numpy.einsum(
                    'iijj->', h2[kp, kp, kq, :nocc, :nocc, :nocc, :nocc]) / nkpts
                Eout -= numpy.einsum(
                    'ijji->', h2[kp, kq, kq, :nocc, :nocc, :nocc, :nocc]) / nkpts
        diff = abs(Eout - Eref)
        self.assertTrue(diff < 1e-12)

        # test in physicists notation
        h1, h2 = integrals.get_solk_integrals(kmf, eriorder='phys')
        h2 = h2/nkpts
        Eout = kmf.energy_nuc()
        for k in range(nkpts):
            Eout += 2.0*numpy.einsum('ii->', h1[k, :nocc, :nocc]) / nkpts

        for kp in range(nkpts):
            for kq in range(nkpts):
                Eout += 2.0*numpy.einsum(
                    'ijij->', h2[kp, kq, kp, :nocc, :nocc, :nocc, :nocc]) / nkpts
                Eout -= numpy.einsum(
                    'ijji->', h2[kp, kq, kq, :nocc, :nocc, :nocc, :nocc]) / nkpts
        diff = abs(Eout - Eref)
        self.assertTrue(diff < 1e-12)


if __name__ == '__main__':
    unittest.main()

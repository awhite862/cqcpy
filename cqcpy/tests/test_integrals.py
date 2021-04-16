import unittest
import numpy


class IntegralsTest(unittest.TestCase):
    def setUp(self):
        from pyscf import gto, scf
        mol = gto.M(
            verbose=0,
            atom='Be 0 0 0',
            basis='sto-3G',
            spin=1, charge=1)
        mf = scf.UHF(mol)
        mf.conv_tol = 1e-13
        mf.scf()
        self.mol = mol
        self.mf = mf

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
        kpt = cell.make_kpts((1,1,1), scaled_center=(0,0,1./3.))
        pbc_mf2 = pbc_scf.RHF(cell, kpt=kpt, exxdiv=None)
        pbc_mf2.kernel()
        self.pbc_mf = pbc_mf
        self.pbc_mf2 = pbc_mf2
        self.cell = cell

    def test_phys(self):
        from cqcpy import integrals
        mo_coeff = self.mf.mo_coeff
        mol = self.mol
        nao = mo_coeff[0].shape[0]
        mo1 = mo_coeff[0][:,0].reshape((nao,1))
        mo2 = mo_coeff[0][:,1].reshape((nao,1))
        mo3 = mo_coeff[1][:,0].reshape((nao,1))
        mo4 = mo_coeff[1][:,1].reshape((nao,1))

        ref = integrals.get_chem(mol, mo1, mo3, mo2, mo4, anti=True).transpose((0,2,1,3))
        out = integrals.get_phys(mol, mo1, mo2, mo3, mo4, anti=True)
        diff = abs(ref[0,0,0,0] - out[0,0,0,0])
        self.assertTrue(diff < 1e-12)

    def test_u(self):
        from cqcpy import integrals
        mo_coeff = self.mf.mo_coeff
        mol = self.mol
        moa = mo_coeff[0]
        mob = mo_coeff[1]
        Iref = integrals.get_chemu(mol, moa, moa, moa, moa, mob, mob, mob, mob, anti=True)
        Iout = integrals.get_chemu_all(mol, moa, mob, anti=True)
        diff = numpy.linalg.norm(Iref - Iout)
        self.assertTrue(diff < 1e-12)

    def test_sol_phys(self):
        from cqcpy import integrals
        mo_coeff = self.pbc_mf.mo_coeff
        nao = mo_coeff.shape[0]
        mo1 = mo_coeff[:,0].reshape((nao,1))
        mo2 = mo_coeff[:,1].reshape((nao,1))
        mo3 = mo_coeff[:,2].reshape((nao,1))
        mo4 = mo_coeff[:,3].reshape((nao,1))

        ref = integrals.get_chem_sol(
            self.pbc_mf, mo1, mo3, mo2, mo4, anti=True).transpose((0,2,1,3))
        out = integrals.get_phys_sol(
            self.pbc_mf, mo1, mo2, mo3, mo4, anti=True)
        diff = abs(ref[0,0,0,0] - out[0,0,0,0])
        self.assertTrue(diff < 1e-12)

    def test_fock(self):
        from cqcpy import integrals
        from cqcpy import utils
        mf = self.mf
        f = mf.get_fock()
        h1 = mf.get_hcore(self.mf.mol)
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
        mf = self.pbc_mf
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
        mf = self.pbc_mf2
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
        mf = self.pbc_mf2
        cc = pbc_cc.CCSD(mf)
        eris = cc.ao2mo()
        mo = mf.mo_coeff
        o = mo[:,mf.mo_occ > 0]
        v = mo[:,mf.mo_occ == 0]

        oooo = eris.oooo
        Ioooo = integrals.get_chem_sol(mf, o, o, o, o)
        diff = numpy.linalg.norm(oooo - Ioooo)
        self.assertTrue(diff < 1e-12)

        oovv = eris.oovv
        Ioovv = integrals.get_chem_sol(mf, o, o, v, v)
        diff = numpy.linalg.norm(oovv - Ioovv)
        self.assertTrue(diff < 1e-12)


if __name__ == '__main__':
    unittest.main()

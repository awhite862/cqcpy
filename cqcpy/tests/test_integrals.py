import unittest
import numpy
from cqcpy import integrals

class IntegralsTest(unittest.TestCase):
    def setUp(self):
        from pyscf import gto, scf
        mol = gto.M(
            verbose = 0,
            atom = 'Be 0 0 0',
            basis = 'sto-3G',
            spin = 1, charge = 1)
        mf = scf.UHF(mol)
        mf.conv_tol = 1e-13
        Escf = mf.scf()
        self.mol = mol
        self.mf = mf

        import pyscf.pbc.gto as pbc_gto
        import pyscf.pbc.scf as pbc_scf
        cell = pbc_gto.Cell()
        cell.atom='''
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
        ehf = pbc_mf.kernel()
        self.pbc_mf = pbc_mf
        self.cell = cell

    def test_phys(self):
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
        mo_coeff = self.mf.mo_coeff
        mol = self.mol
        moa = mo_coeff[0]
        mob = mo_coeff[1]
        Iref = integrals.get_chemu(mol, moa, moa, moa, moa, mob, mob, mob, mob, anti=True)
        Iout = integrals.get_chemu_all(mol, moa, mob, anti=True)
        diff = numpy.linalg.norm(Iref - Iout)
        self.assertTrue(diff < 1e-12)

    def test_sol_phys(self):
        mo_coeff = self.pbc_mf.mo_coeff
        nao = mo_coeff.shape[0]
        mo1 = mo_coeff[:,0].reshape((nao,1))
        mo2 = mo_coeff[:,1].reshape((nao,1))
        mo3 = mo_coeff[:,2].reshape((nao,1))
        mo4 = mo_coeff[:,3].reshape((nao,1))

        ref = integrals.get_chem_sol(self.pbc_mf, mo1, mo3, mo2, mo4, anti=True).transpose((0,2,1,3))
        out = integrals.get_phys_sol(self.pbc_mf, mo1, mo2, mo3, mo4, anti=True)
        diff = abs(ref[0,0,0,0] - out[0,0,0,0])
        self.assertTrue(diff < 1e-12)

if __name__ == '__main__':
    unittest.main()

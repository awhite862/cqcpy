import unittest
import numpy

from cqcpy import integrals
from cqcpy import ci_utils


class CIUtilsTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_mat_on_vec_cisd(self):
        nmo = 9
        na = 2
        nb = 2
        thresh = 1e-14

        ha = numpy.random.random((nmo, nmo))
        ha = ha + ha.transpose((1, 0))
        hb = ha
        Ia = numpy.random.random((nmo, nmo, nmo, nmo))
        Ia = Ia + Ia.transpose((1, 0, 3, 2))
        Ib = Ia
        Iabab = Ia

        sa = ci_utils.s_strings(nmo, na)
        sb = ci_utils.s_strings(nmo, nb)
        da = ci_utils.d_strings(nmo, na)
        db = ci_utils.d_strings(nmo, nb)
        occ = [1 if i < na else 0 for i in range(nmo)]
        ref = ci_utils.Dstring(nmo, occ)
        basis = []

        # scf ground state
        basis.append((ref, ref))

        # singly excited states
        for a in sa:
            basis.append((a, ref))
        for b in sb:
            basis.append((ref, b))

        # doubly excited states
        for a in da:
            basis.append((a, ref))
        for b in db:
            basis.append((ref, b))
        for a in sa:
            for b in sb:
                basis.append((a, b))

        nd = len(basis)
        vec = numpy.random.random(nd)
        H = numpy.zeros((nd, nd))
        for i, b in enumerate(basis):
            for j, k in enumerate(basis):
                H[i, j] = ci_utils.ci_matrixel(
                    b[0], b[1], k[0], k[1], ha, hb, Ia, Ib, Iabab, 0.0)
        ref = numpy.einsum('ij,j->i', H, vec)
        out = ci_utils.H_on_vec(basis, vec, ha, hb, Ia, Ib, Iabab)
        diff = numpy.linalg.norm(out - ref)/numpy.linalg.norm(ref)
        self.assertTrue(diff < thresh, "Difference: {}".format(diff))

    def test_cis(self):
        from pyscf import gto
        from pyscf.tdscf.rhf import scf

        # reference
        mol = gto.Mole()
        mol.build(
            atom='H 0 0 0; H 0 0 0.77',  # in Angstrom
            basis='631g', verbose=0)

        mf = scf.RHF(mol)
        mf.conv_tol_grad = 1e-9
        Escf = mf.kernel()
        mf.verbose = 0
        tda = mf.TDA()
        tda.nstates = 6
        erefs = tda.kernel()[0]
        tda.singlet = False
        ereft = tda.kernel()[0]

        # run CIS calculation
        mos = mf.mo_coeff
        nmo = mos.shape[1]
        hcore = mf.get_hcore()
        ha = numpy.einsum('mp,mn,nq->pq', mos, hcore, mos)
        hb = ha.copy()
        I = integrals.get_phys(mol, mos, mos, mos, mos)

        N = mol.nelectron
        na = N//2
        nb = na
        basis = ci_utils.ucis_basis(nmo, na, nb, gs=False)
        nd = len(basis)

        H = numpy.zeros((nd, nd))
        const = -Escf + mol.energy_nuc()
        F = mf.get_fock()
        F = numpy.einsum('mp,mn,nq->pq', mos, F, mos)
        for i, b in enumerate(basis):
            for j, k in enumerate(basis):
                H[i, j] = ci_utils.ci_matrixel(
                    b[0], b[1], k[0], k[1], ha, hb, I, I, I, const)

        eout, v = numpy.linalg.eigh(H)
        eref = numpy.sort(numpy.concatenate((erefs, ereft)))
        diff = numpy.linalg.norm(eref - eout)
        self.assertTrue(diff < 1e-10)

    def test_cisd(self):
        from pyscf import gto, scf, ci

        mol = gto.M(
            atom='Be 0 0 0',
            basis='631g', verbose=0)
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        Escf = mf.kernel()
        myci = ci.CISD(mf).run()
        #print('RCISD correlation energy', myci.e_corr)
        ref = myci.e_corr

        # run naive CISD calculation
        mos = mf.mo_coeff
        nmo = mos.shape[1]
        hcore = mf.get_hcore()
        ha = numpy.einsum('mp,mn,nq->pq', mos, hcore, mos)
        hb = ha.copy()
        I = integrals.get_phys(mol, mos, mos, mos, mos)

        N = mol.nelectron
        na = N//2
        nb = na

        basis = ci_utils.ucisd_basis(nmo, na, nb)
        nd = len(basis)
        H = numpy.zeros((nd, nd))
        const = -Escf + mol.energy_nuc()
        for i, b in enumerate(basis):
            for j, k in enumerate(basis):
                H[i, j] = ci_utils.ci_matrixel(
                    b[0], b[1], k[0], k[1], ha, hb, I, I, I, const)

        eout, v = numpy.linalg.eigh(H)
        #print('RCISD correlation energy', eout[0])
        diff = abs(ref - eout[0])
        self.assertTrue(diff < 1e-12)


if __name__ == '__main__':
    unittest.main()

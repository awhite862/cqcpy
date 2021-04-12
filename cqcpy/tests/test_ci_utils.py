import unittest
import numpy

from cqcpy import ci_utils

class CIUtilsTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_mat_on_vec_cisd(self):
        nmo = 9
        na = 2
        nb = 2
        thresh = 1e-14

        ha = numpy.random.random((nmo,nmo))
        ha = ha + ha.transpose((1,0))
        hb = ha
        Ia = numpy.random.random((nmo,nmo,nmo,nmo))
        Ia = Ia + Ia.transpose((1,0,3,2))
        Ib = Ia
        Iabab = Ia

        sa = ci_utils.s_strings(nmo, na)
        sb = ci_utils.s_strings(nmo, nb)
        da = ci_utils.d_strings(nmo, na)
        db = ci_utils.d_strings(nmo, nb)
        occ = [1 if i < na else 0 for i in range(nmo)]
        ref = ci_utils.Dstring(nmo,occ)
        basis = []

        # scf ground state
        basis.append((ref,ref))

        # singly excited states
        for a in sa:
            basis.append((a,ref))
        for b in sb:
            basis.append((ref,b))

        # doubly excited states
        for a in da:
            basis.append((a,ref))
        for b in db:
            basis.append((ref,b))
        for a in sa:
            for b in sb:
                basis.append((a,b))

        nd = len(basis)
        vec = numpy.random.random(nd)
        H = numpy.zeros((nd,nd))
        for i,b in enumerate(basis):
            for j,k in enumerate(basis):
                H[i,j] = ci_utils.ci_matrixel(b[0],b[1],k[0],k[1],ha,hb,Ia,Ib,Iabab,0.0)
        ref = numpy.einsum('ij,j->i',H,vec)
        out = ci_utils.H_on_vec(basis, vec, ha, hb, Ia, Ib, Iabab)
        diff = numpy.linalg.norm(out - ref)/numpy.linalg.norm(ref)
        self.assertTrue(diff < thresh,"Difference: {}".format(diff))

if __name__ == '__main__':
    unittest.main()

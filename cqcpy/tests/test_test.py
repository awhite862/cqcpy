import unittest
import numpy
from cqcpy import test_utils


class TestTest(unittest.TestCase):
    def setUp(self):
        self.thresh = 1e-14

    def test_framework(self):
        self.assertTrue(True)

    def test_Isym(self):
        I = test_utils.make_random_Itot(5)
        test = I - I.transpose((1,0,3,2))
        s1 = numpy.linalg.norm(test) < self.thresh
        test = I - I.transpose((3,2,1,0))
        s2 = numpy.linalg.norm(test) < self.thresh
        test = I - I.transpose((2,3,0,1))
        s3 = numpy.linalg.norm(test) < self.thresh
        err = "Bad symmetry in full I"
        self.assertTrue(s1 and s2 and s3, err)

    def test_int_sym(self):
        F,I = test_utils.make_random_integrals(2, 3)
        test = F.oo - F.oo.transpose((1,0))
        sym = numpy.linalg.norm(test) < self.thresh
        err = "Bad symmetry in Foo"
        self.assertTrue(sym, err)
        test = F.vv - F.vv.transpose((1,0))
        sym = numpy.linalg.norm(test) < self.thresh
        err = "Bad symmetry in Fvv"
        self.assertTrue(sym, err)
        test = F.ov - F.vo.transpose((1,0))
        sym = numpy.linalg.norm(test) < self.thresh
        err = "Bad symmetry in Fov/Fvo"
        self.assertTrue(sym, err)

        test = I.vvvv + I.vvvv.transpose((0,1,3,2))
        sym1 = numpy.linalg.norm(test) < self.thresh
        test = I.vvvv + I.vvvv.transpose((1,0,2,3))
        sym2 = numpy.linalg.norm(test) < self.thresh
        test = I.vvvv - I.vvvv.transpose((1,0,3,2))
        sym3 = numpy.linalg.norm(test) < self.thresh
        err = "Bad symmetry in Ivvvv"
        self.assertTrue(sym1 and sym2 and sym3, err)

        test = I.vvvo + I.vvvo.transpose((1,0,2,3))
        sym = numpy.linalg.norm(test) < self.thresh
        err = "Bad symmetry in Ivvvo"
        self.assertTrue(sym, err)

        test = I.vovv + I.vovv.transpose((0,1,3,2))
        sym = numpy.linalg.norm(test) < self.thresh
        err = "Bad symmetry in Ivovv"
        self.assertTrue(sym, err)

        test = I.vvoo + I.vvoo.transpose((0,1,3,2))
        sym1 = numpy.linalg.norm(test) < self.thresh
        test = I.vvoo + I.vvoo.transpose((1,0,2,3))
        sym2 = numpy.linalg.norm(test) < self.thresh
        err = "Bad symmetry in Ivvoo"
        self.assertTrue(sym1 and sym2, err)

        test = I.oovv + I.oovv.transpose((0,1,3,2))
        sym1 = numpy.linalg.norm(test) < self.thresh
        test = I.oovv + I.oovv.transpose((1,0,2,3))
        sym2 = numpy.linalg.norm(test) < self.thresh
        err = "Bad symmetry in Ioovv"
        self.assertTrue(sym1 and sym2, err)

        test = I.ooov + I.ooov.transpose((1,0,2,3))
        sym = numpy.linalg.norm(test) < self.thresh
        err = "Bad symmetry in Iooov"
        self.assertTrue(sym, err)

        test = I.oooo + I.oooo.transpose((0,1,3,2))
        sym1 = numpy.linalg.norm(test) < self.thresh
        test = I.oooo + I.oooo.transpose((1,0,2,3))
        sym2 = numpy.linalg.norm(test) < self.thresh
        test = I.oooo - I.oooo.transpose((1,0,3,2))
        sym3 = numpy.linalg.norm(test) < self.thresh
        err = "Bad symmetry in Ioooo"
        self.assertTrue(sym1 and sym2 and sym3, err)

    def test_Tsym(self):
        no = 3
        nv = 4
        T1, T2 = test_utils.make_random_T(no, nv)

        test = T2 + T2.transpose((0,1,3,2))
        s1 = numpy.linalg.norm(test) < self.thresh
        test = T2 + T2.transpose((1,0,2,3))
        s2 = numpy.linalg.norm(test) < self.thresh
        test = T2 - T2.transpose((1,0,3,2))
        s3 = numpy.linalg.norm(test) < self.thresh
        err = "Bad symmetry in T2"
        self.assertTrue(s1 and s2 and s3, err)

    def test_Lsym(self):
        no = 3
        nv = 4
        L1, L2 = test_utils.make_random_L(no, nv)

        test = L2 + L2.transpose((0,1,3,2))
        s1 = numpy.linalg.norm(test) < self.thresh
        test = L2 + L2.transpose((1,0,2,3))
        s2 = numpy.linalg.norm(test) < self.thresh
        test = L2 - L2.transpose((1,0,3,2))
        s3 = numpy.linalg.norm(test) < self.thresh
        err = "Bad symmetry in T2"
        self.assertTrue(s1 and s2 and s3, err)

    def test_ft_int_sym(self):
        F, I = test_utils.make_random_ft_integrals(5)
        test = I + I.transpose((0,1,3,2))
        sym1 = numpy.linalg.norm(test) < self.thresh
        test = I + I.transpose((1,0,2,3))
        sym2 = numpy.linalg.norm(test) < self.thresh
        test = I - I.transpose((1,0,3,2))
        sym3 = numpy.linalg.norm(test) < self.thresh
        err = "Bad symmetry in FT I"
        self.assertTrue(sym1 and sym2 and sym3, err)

        test = F - F.transpose((1,0))
        s = numpy.linalg.norm(test) < self.thresh
        err = "Bad symmetry in FT Fock"
        self.assertTrue(s, err)

    def test_ft_Tsym(self):
        T1, T2 = test_utils.make_random_ft_T(5, 3)
        test = T2 + T2.transpose((0,1,2,4,3))
        sym1 = numpy.linalg.norm(test) < self.thresh
        test = T2 + T2.transpose((0,2,1,3,4))
        sym2 = numpy.linalg.norm(test) < self.thresh
        test = T2 - T2.transpose((0,2,1,4,3))
        sym3 = numpy.linalg.norm(test) < self.thresh
        err = "Bad symmetry in FT I"
        self.assertTrue(sym1 and sym2 and sym3, err)


if __name__ == '__main__':
    unittest.main()

import unittest
import numpy
from cqcpy import ft_utils


def _fd_grand_potential(beta, epsilon, mu, delta=1e-4):
    fff = ft_utils.grand_potential0(beta + delta, epsilon, mu)
    bbb = ft_utils.grand_potential0(beta - delta, epsilon, mu)
    ref = (fff - bbb)/(2.0*delta)
    return ref


class FTUtilsTest(unittest.TestCase):
    def setUp(self):
        self.betas = [0.01, 0.1, 1.0, 10.0]
        self.es = [-10.0, -1.0, 0.0, 1.0, 10.0]
        self.Fs = [[
            0.524979187478939986,
            0.502499979166874998,
            0.500000000000000000,
            0.497500020833125002,
            0.475020812521060014],
            [
            0.731058578630004879,
            0.524979187478939986,
            0.500000000000000000,
            0.475020812521060014,
            0.268941421369995121],
            [
            0.999954602131297566,
            0.731058578630004879,
            0.500000000000000000,
            0.268941421369995121,
            0.453978687024343945e-4],
            [
            1.00000000000000000,
            0.999954602131297566,
            0.500000000000000000,
            0.453978687024343945e-4,
            0.372007597602083596e-43]]

        self.beta = 2.0
        self.thresh = 1e-7

    def test_fermi(self):
        tol = 1e-14
        for i, b in enumerate(self.betas):
            for j, e in enumerate(self.es):
                val = ft_utils.fermi_function(b, e, 0.0)
                ref = self.Fs[i][j]
                fail = abs(val - ref)/abs(ref) > tol
                self.assertFalse(fail, "Value: {}  Ref: {}".format(val, ref))
        for i, b in enumerate(self.betas):
            for j, e in enumerate(self.es):
                val = ft_utils.fermi_function(b, e + 0.1, 0.1)
                ref = self.Fs[i][j]
                fail = abs(val - ref)/abs(ref) > tol
                self.assertFalse(fail, "Value: {}  Ref: {}".format(val, ref))

    def test_vfermi(self):
        tol = 1e-14
        for i, b in enumerate(self.betas):
            for j, e in enumerate(self.es):
                val = ft_utils.vfermi_function(b, e, 0.0)
                ref = self.Fs[i][4 - j]
                fail = abs(val - ref)/abs(ref) > tol
                self.assertFalse(fail, "Value: {}  Ref: {}".format(val, ref))
        for i, b in enumerate(self.betas):
            for j, e in enumerate(self.es):
                val = ft_utils.vfermi_function(b, e + 0.1, 0.1)
                ref = self.Fs[i][4 - j]
                fail = abs(val - ref)/abs(ref) > tol
                self.assertFalse(fail, "Value: {}  Ref: {}".format(val, ref))

    def test_dfermi(self):
        beta = 2.0
        e = 1.0
        mu = 0.7
        delta = 1e-4
        fo = ft_utils.fermi_function(beta, e, mu)
        fv = ft_utils.vfermi_function(beta, e, mu)
        fof = numpy.sqrt(ft_utils.fermi_function(beta + delta, e, mu))
        fob = numpy.sqrt(ft_utils.fermi_function(beta - delta, e, mu))
        fvf = numpy.sqrt(ft_utils.vfermi_function(beta + delta, e, mu))
        fvb = numpy.sqrt(ft_utils.vfermi_function(beta - delta, e, mu))
        dfo_fd = (fof - fob)/(2.0*delta)
        dfv_fd = (fvf - fvb)/(2.0*delta)
        dx = (e - mu)
        dfo = -0.5*dx*fv*numpy.sqrt(fo)
        dfv = 0.5*dx*numpy.sqrt(fv)*fo
        diffo = abs(dfo - dfo_fd)
        diffv = abs(dfv - dfv_fd)
        self.assertTrue(diffo < 1e-8)
        self.assertTrue(diffv < 1e-8)

    def test_vector(self):
        beta = 1.0
        mu = 0.01
        es = numpy.asarray([-1, -0.5, 0.1, 0.5, 1])
        out = ft_utils.ff(beta, es, mu)
        ref = numpy.asarray(
            [ft_utils.fermi_function(beta, e, mu) for e in es])
        diff = numpy.linalg.norm(ref - out)
        self.assertTrue(diff < 5e-14)

        out = ft_utils.ffv(beta, es, mu)
        ref = numpy.asarray(
            [ft_utils.vfermi_function(beta, e, mu) for e in es])
        diff = numpy.linalg.norm(ref - out)
        self.assertTrue(diff < 5e-14)

    def test_dgrand_potentiala(self):
        beta = 3.0
        e = 1.0
        mu = 0.7
        ref = _fd_grand_potential(beta, e, mu)
        out = ft_utils.dgrand_potential0(beta, e, mu)
        diff = abs(out - ref)/abs(ref)
        self.assertTrue(diff < 1e-8)

    def test_dgrand_potentialb(self):
        beta = 3.0
        e = 1.0
        mu = -6.0
        ref = _fd_grand_potential(beta, e, mu)
        out = ft_utils.dgrand_potential0(beta, e, mu)
        diff = abs(out - ref)/abs(ref)
        self.assertTrue(diff < 1e-6)

    def test_dgrand_potentialc(self):
        beta = 3.0
        e = -6.0
        mu = 1.0
        ref = _fd_grand_potential(beta, e, mu, delta=6e-4)
        out = ft_utils.dgrand_potential0(beta, e, mu)
        diff = abs(out - ref)/abs(ref)
        self.assertTrue(diff < 1e-3)

    def test_dGP0(self):
        beta = 3.0
        e = numpy.asarray([-1.0,  -0.07, 0.6, 2.1])
        mu = 0.7
        delta = 1e-4
        fff = ft_utils.GP0(beta + delta, e, mu)
        bbb = ft_utils.GP0(beta - delta, e, mu)
        ref = (fff - bbb)/(2.0*delta)
        out = ft_utils.dGP0(beta, e, mu)
        diff = numpy.linalg.norm(out - ref)/numpy.linalg.norm(ref)
        self.assertTrue(diff < 1e-8)


if __name__ == '__main__':
    unittest.main()

import unittest
import numpy
from cqcpy import test_utils
from cqcpy import ft_utils

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
        for i,b in enumerate(self.betas):
            for j,e in enumerate(self.es):
                val = ft_utils.fermi_function(b,e,0.0)
                ref = self.Fs[i][j]
                fail = abs(val - ref)/abs(ref) > tol
                self.assertFalse(fail,"Value: {}  Ref: {}".format(val,ref))
        for i,b in enumerate(self.betas):
            for j,e in enumerate(self.es):
                val = ft_utils.fermi_function(b,e + 0.1,0.1)
                ref = self.Fs[i][j]
                fail = abs(val - ref)/abs(ref) > tol
                self.assertFalse(fail,"Value: {}  Ref: {}".format(val,ref))

    def test_vfermi(self):
        tol = 1e-14
        for i,b in enumerate(self.betas):
            for j,e in enumerate(self.es):
                val = ft_utils.vfermi_function(b,e,0.0)
                ref = self.Fs[i][4 - j]
                fail = abs(val - ref)/abs(ref) > tol
                self.assertFalse(fail,"Value: {}  Ref: {}".format(val,ref))
        for i,b in enumerate(self.betas):
            for j,e in enumerate(self.es):
                val = ft_utils.vfermi_function(b,e + 0.1,0.1)
                ref = self.Fs[i][4 - j]
                fail = abs(val - ref)/abs(ref) > tol
                self.assertFalse(fail,"Value: {}  Ref: {}".format(val,ref))


if __name__ == '__main__':
    unittest.main()

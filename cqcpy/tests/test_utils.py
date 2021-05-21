import unittest
import numpy
from cqcpy import utils


class UtilsTest(unittest.TestCase):
    def test_block_diag(self):
        A = numpy.zeros((2, 2))
        A[0, 0] = 13.4
        A[0, 1] = 19
        A[1, 0] = 25
        A[1, 1] = -50
        B = numpy.zeros((4, 2))
        B[0, 0] = -48
        B[0, 1] = 21.8
        B[1, 0] = -100
        B[2, 1] = -44
        B[3, 0] = 50
        ref = numpy.zeros((6, 4))
        ref[0, 0] = A[0, 0]
        ref[0, 1] = A[0, 1]
        ref[1, 0] = A[1, 0]
        ref[1, 1] = A[1, 1]
        ref[2, 2] = B[0, 0]
        ref[2, 3] = B[0, 1]
        ref[3, 2] = B[1, 0]
        ref[4, 3] = B[2, 1]
        ref[5, 2] = B[3, 0]
        out = utils.block_diag(A, B)
        self.assertTrue(numpy.linalg.norm(ref - out) < 1e-14)

    def test_D1(self):
        eo = numpy.asarray([-10.1, -9.2356, -5.2, -2])
        ev = numpy.asarray([0, 4, 5.2, 5.9, 7, 13, 21.244])
        no, nv = len(eo), len(ev)
        D1ref = numpy.zeros((nv, no))
        for a in range(nv):
            for i in range(no):
                D1ref[a, i] = ev[a] - eo[i]

        D1out = utils.D1(ev, eo)
        diff = numpy.linalg.norm(D1ref - D1out)
        self.assertTrue(diff < 1e-12)

    def test_D2(self):
        eo = numpy.asarray([-10.1, -9.2356, -5.2, -2])
        ev = numpy.asarray([0, 4, 5.2, 5.9, 7, 13, 21.244])
        no, nv = len(eo), len(ev)
        D2ref = numpy.zeros((nv, nv, no, no))
        for a in range(nv):
            for b in range(nv):
                for i in range(no):
                    for j in range(no):
                        D2ref[a, b, i, j] = ev[a] + ev[b] - eo[i] - eo[j]

        D2out = utils.D2(ev, eo)
        diff = numpy.linalg.norm(D2ref - D2out)
        self.assertTrue(diff < 1e-12)


if __name__ == '__main__':
    unittest.main()

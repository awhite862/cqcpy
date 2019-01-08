import numpy

def block_diag(A,B):
    """Return a block diagonal matrix
       A 0
       0 B
    """
    ma = A.shape[0]
    mb = B.shape[0]
    na = A.shape[1]
    nb = B.shape[1]

    z1 = numpy.zeros((ma,nb))
    z2 = numpy.zeros((mb,na))
    M1 = numpy.hstack((A,z1))
    M2 = numpy.hstack((z2,B))
    return numpy.vstack((M1,M2))

import numpy
from .ov_blocks import one_e_blocks
from .ov_blocks import two_e_blocks
from .ov_blocks import two_e_blocks_full
from .ov_blocks import make_two_e_blocks_full


def make_random_F(no, nv):
    foo = numpy.random.random((no,no))
    fov = numpy.random.random((no,nv))
    fvv = numpy.random.random((nv,nv))
    foo += foo.transpose((1,0))
    fvv += fvv.transpose((1,0))
    fvo = fov.transpose((1,0))
    return one_e_blocks(foo, fov, fvo, fvv)


def make_random_I(no, nv):
    n = no + nv
    Itot = numpy.random.random((n,n,n,n))
    Itot += Itot.transpose((1,0,3,2))
    Ivvvv = Itot[no:,no:,no:,no:]
    Ivvvo = Itot[no:,no:,no:,:no]
    Ivovv = Itot[no:,:no,no:,no:]
    Ivvoo = Itot[no:,no:,:no,:no]
    Ivovo = Itot[no:,:no,no:,:no]
    Ioovv = Itot[:no,:no,no:,no:]
    Ivooo = Itot[no:,:no,:no,:no]
    Iooov = Itot[:no,:no,:no,no:]
    Ioooo = Itot[:no,:no,:no,:no]
    return two_e_blocks(
        vvvv=Ivvvv, vvvo=Ivvvo, vovv=Ivovv, vvoo=Ivvoo,
        vovo=Ivovo, oovv=Ioovv, vooo=Ivooo, ooov=Iooov, oooo=Ioooo)


def make_random_I_anti(no, nv):
    n = no + nv
    Itot = numpy.random.random((n,n,n,n))
    Itot -= Itot.transpose((0,1,3,2))
    Itot -= Itot.transpose((1,0,2,3))
    Itot += Itot.transpose((1,0,3,2))
    Ivvvv = Itot[no:,no:,no:,no:]
    Ivvvo = Itot[no:,no:,no:,:no]
    Ivovv = Itot[no:,:no,no:,no:]
    Ivvoo = Itot[no:,no:,:no,:no]
    Ivovo = Itot[no:,:no,no:,:no]
    Ioovv = Itot[:no,:no,no:,no:]
    Ivooo = Itot[no:,:no,:no,:no]
    Iooov = Itot[:no,:no,:no,no:]
    Ioooo = Itot[:no,:no,:no,:no]
    return two_e_blocks(
        vvvv=Ivvvv, vvvo=Ivvvo, vovv=Ivovv, vvoo=Ivvoo,
        vovo=Ivovo, oovv=Ioovv, vooo=Ivooo, ooov=Iooov, oooo=Ioooo)


def make_random_Itot(n):
    I = numpy.random.random((n,n,n,n))
    I += I.transpose((1,0,3,2))
    I += I.transpose((2,3,0,1))
    I += I.transpose((3,2,1,0))

    return I


def make_random_Ifull(no, nv):
    n = no + nv
    Itot = numpy.random.random((n,n,n,n))
    Itot -= Itot.transpose((0,1,3,2))
    Itot -= Itot.transpose((1,0,2,3))
    Itot += Itot.transpose((1,0,3,2))
    Ivvvv = Itot[no:,no:,no:,no:]
    Ivvvo = Itot[no:,no:,no:,:no]
    Ivvov = Ivvvo.transpose((1,0,3,2))
    Ivovv = Itot[no:,:no,no:,no:]
    Iovvv = Ivovv.transpose((1,0,3,2))
    Ivvoo = Itot[no:,no:,:no,:no]
    Ivovo = Itot[no:,:no,no:,:no]
    Iovov = Ivovo.transpose((1,0,3,2))
    Iovvo = Itot[:no,no:,no:,:no]
    Ivoov = Iovvo.transpose((1,0,3,2))
    Ioovv = Itot[:no,:no,no:,no:]
    Ivooo = Itot[no:,:no,:no,:no]
    Iovoo = Ivooo.transpose((1,0,3,2))
    Iooov = Itot[:no,:no,:no,no:]
    Ioovo = Iooov.transpose((1,0,3,2))
    Ioooo = Itot[:no,:no,:no,:no]
    return two_e_blocks_full(
        vvvv=Ivvvv, vvvo=Ivvvo, vvov=Ivvov, vovv=Ivovv,
        ovvv=Iovvv, vvoo=Ivvoo, vovo=Ivovo, ovov=Iovov,
        voov=Ivoov, ovvo=Iovvo, oovv=Ioovv, vooo=Ivooo,
        ovoo=Iovoo, oovo=Ioovo, ooov=Iooov, oooo=Ioooo)


def make_random_Ifull_gen(n1o, n1v, n2o, n2v, n3o, n3v, n4o, n4v):
    n1 = n1o + n1v
    n2 = n2o + n2v
    n3 = n3o + n3v
    n4 = n4o + n4v
    Itot = numpy.random.random((n1,n2,n3,n4))
    return make_two_e_blocks_full(Itot, n1o, n1v, n2o, n2v, n3o, n3v, n4o, n4v)


def make_random_integrals(no, nv):
    F = make_random_F(no, nv)
    I = make_random_I_anti(no, nv)

    return F, I


def make_random_T(no, nv):
    T1 = numpy.random.random((nv,no))
    T2 = numpy.random.random((nv,nv,no,no))
    T2 -= T2.transpose((1,0,2,3))
    T2 -= T2.transpose((0,1,3,2))
    T2 += T2.transpose((1,0,3,2))

    return T1, T2


def make_random_T1_spatial(noa, nva, nob, nvb):
    Ta = numpy.random.random((nva,noa))
    Tb = numpy.random.random((nvb,nob))
    return Ta, Tb


def make_random_T2_spatial(noa, nva, nob, nvb):
    Taa = numpy.random.random((nva,nva,noa,noa))
    Tbb = numpy.random.random((nvb,nvb,nob,nob))
    Tab = numpy.random.random((nva,nvb,noa,nob))
    Taa -= Taa.transpose((1,0,2,3))
    Taa -= Taa.transpose((0,1,3,2))
    Taa += Taa.transpose((1,0,3,2))
    Tbb -= Tbb.transpose((1,0,2,3))
    Tbb -= Tbb.transpose((0,1,3,2))
    Tbb += Tbb.transpose((1,0,3,2))
    return Taa, Tab, Tbb


def make_random_L(no, nv):
    L1 = numpy.random.random((no,nv))
    L2 = numpy.random.random((no,no,nv,nv))
    L2 -= L2.transpose((1,0,2,3))
    L2 -= L2.transpose((0,1,3,2))
    L2 += L2.transpose((1,0,3,2))

    return L1, L2


def make_random_ft_integrals(n):
    F = numpy.random.random((n,n))
    F += F.transpose((1,0))

    I = numpy.random.random((n,n,n,n))
    I -= I.transpose((0,1,3,2))
    I -= I.transpose((1,0,2,3))
    I += I.transpose((1,0,3,2))

    return F, I


def make_random_ft_T(ng, n):
    # Note this also can be used for Lambdas
    T1 = numpy.random.random((ng,n,n))
    T2 = numpy.random.random((ng,n,n,n,n))
    T2 -= T2.transpose((0,1,2,4,3))
    T2 -= T2.transpose((0,2,1,3,4))
    T2 += T2.transpose((0,2,1,4,3))

    return T1, T2


def make_random_ft_T1_spatial(ng, na, nb):
    # Note this also can be used for Lambdas
    T1a = numpy.random.random((ng,na,na))
    T1b = numpy.random.random((ng,nb,nb))

    return T1a, T1b


def make_random_ft_T2_spatial(ng, na, nb):
    T2aa = numpy.random.random((ng,na,na,na,na))
    T2aa -= T2aa.transpose((0,1,2,4,3))
    T2aa -= T2aa.transpose((0,2,1,3,4))
    T2aa += T2aa.transpose((0,2,1,4,3))
    T2bb = numpy.random.random((ng,nb,nb,nb,nb))
    T2bb -= T2bb.transpose((0,1,2,4,3))
    T2bb -= T2bb.transpose((0,2,1,3,4))
    T2bb += T2bb.transpose((0,2,1,4,3))
    T2ab = numpy.random.random((ng,na,nb,na,nb))
    return T2aa, T2ab, T2bb



def make_random_ft_D(n):
    en = numpy.random.random((n))
    D1 = en[:,None] - en[None,:]
    D2 = en[:,None,None,None] + en[None,:,None,None] \
            - en[None,None,:,None] - en[None,None,None,:]

    return D1, D2


def make_random_ft_D1(n):
    en = numpy.random.random((n))
    D1 = en[:,None] - en[None,:]

    return D1


def make_random_ft_D2(n1, n2):
    e1 = numpy.random.random((n1))
    e2 = numpy.random.random((n2))
    D2 = e1[:,None,None,None] + e2[None,:,None,None] \
            - e1[None,None,:,None] - e2[None,None,None,:]

    return D2

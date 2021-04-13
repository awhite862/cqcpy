import numpy
from cqcpy.ov_blocks import one_e_blocks
from cqcpy.ov_blocks import two_e_blocks
from cqcpy.ov_blocks import two_e_blocks_full

def F_to_spin(Faa, Fbb, noa, nva, nob, nvb):
    no = noa + nob
    nv = nva + nvb
    Foo = numpy.zeros((no,no))
    Foo[:noa,:noa] = Faa.oo
    Foo[noa:,noa:] = Fbb.oo

    Fov = numpy.zeros((no,nv))
    Fov[:noa,:nva] = Faa.ov
    Fov[noa:,nva:] = Fbb.ov

    Fvo = numpy.zeros((nv,no))
    Fvo[:nva,:noa] = Faa.vo
    Fvo[nva:,noa:] = Fbb.vo

    Fvv = numpy.zeros((nv,nv))
    Fvv[:nva,:nva] = Faa.vv
    Fvv[nva:,nva:] = Fbb.vv

    return one_e_blocks(Foo, Fov, Fvo, Fvv)

def F_to_spatial(F, noa, nva, nob, nvb):

    # Fa
    oo = F.oo[:noa,:noa]
    ov = F.ov[:noa,:nva]
    vo = F.vo[:nva,:noa]
    vv = F.vv[:nva,:nva]
    Fa = one_e_blocks(oo, ov, vo, vv)

    # Fb
    oo = F.oo[noa:,noa:]
    ov = F.ov[noa:,nva:]
    vo = F.vo[nva:,noa:]
    vv = F.vv[nva:,nva:]
    Fb = one_e_blocks(oo, ov, vo, vv)

    return Fa, Fb

def T1_to_spin(Ta, Tb, noa, nva, nob, nvb):
    no = noa + nob
    nv = nva + nvb
    T = numpy.zeros((nv,no))
    T[:nva,:noa] = Ta.copy()
    T[nva:,noa:] = Tb.copy()
    return T

def T1_to_spatial(T, noa, nva, nob, nvb):
    Ta = T[:nva,:noa].copy()
    Tb = T[nva:,noa:].copy()
    return Ta,Tb

def T2u(T2, noa, nva, nob, nvb):
    aaab = T2[:nva,:nva,:noa,noa:]
    aaba = T2[:nva,:nva,noa:,:noa]
    abaa = T2[:nva,nva:,:noa,:noa]
    baaa = T2[nva:,:nva,:noa,:noa]
    aabb = T2[:nva,:nva,noa:,noa:]
    bbaa = T2[nva:,nva:,:noa,:noa]
    bbba = T2[nva:,nva:,noa:,:noa]
    bbab = T2[nva:,nva:,:noa,noa:]
    babb = T2[nva:,:nva,noa:,noa:]
    abbb = T2[:nva,nva:,noa:,noa:]
    print(numpy.linalg.norm(aaab))
    print(numpy.linalg.norm(aaba))
    print(numpy.linalg.norm(abaa))
    print(numpy.linalg.norm(baaa))
    print(numpy.linalg.norm(aabb))
    print(numpy.linalg.norm(bbaa))
    print(numpy.linalg.norm(bbba))
    print(numpy.linalg.norm(bbab))
    print(numpy.linalg.norm(babb))
    print(numpy.linalg.norm(abbb))


def T2_to_spin(Taa, Tab, Tbb, noa, nva, nob, nvb):
    no = noa + nob
    nv = nva + nvb
    T = numpy.zeros((nv,nv,no,no))
    T[:nva,:nva,:noa,:noa] = Taa.copy() # aaaa
    T[nva:,nva:,noa:,noa:] = Tbb.copy() # bbbb
    T[:nva,nva:,:noa,noa:] = Tab.copy() # abab
    T[nva:,:nva,noa:,:noa] = Tab.transpose((1,0,3,2)).copy() # baba
    T[:nva,nva:,noa:,:noa] = -Tab.transpose((0,1,3,2)).copy() # abba
    T[nva:,:nva,:noa,noa:] = -Tab.transpose((1,0,2,3)).copy() # baab
    return T

def D2_to_spin(Daa, Dab, Dbb, noa, nva, nob, nvb):
    no = noa + nob
    nv = nva + nvb
    D = numpy.zeros((nv,nv,no,no))
    D[:nva,:nva,:noa,:noa] = Daa.copy() # aaaa
    D[nva:,nva:,noa:,noa:] = Dbb.copy() # bbbb
    D[:nva,nva:,:noa,noa:] = Dab.copy() # abab
    D[nva:,:nva,noa:,:noa] = Dab.transpose((1,0,3,2)).copy() # baba
    D[:nva,nva:,noa:,:noa] = Dab.transpose((0,1,3,2)).copy() # abba
    D[nva:,:nva,:noa,noa:] = Dab.transpose((1,0,2,3)).copy() # baab
    return D

def T2_to_spatial(T, noa, nva, nob, nvb):
    Taa = T[:nva,:nva,:noa,:noa].copy()
    Tbb = T[nva:,nva:,noa:,noa:].copy()
    Tab = T[:nva,nva:,:noa,noa:].copy()
    return (Taa,Tab,Tbb)

def int_to_spin_block(aaaa, bbbb, abab, baba, abba, baab,
        na1, na2, na3, na4, nb1, nb2, nb3, nb4):
    n1 = na1 + nb1
    n2 = na2 + nb2
    n3 = na3 + nb3
    n4 = na4 + nb4
    I = numpy.zeros((n1,n2,n3,n4))
    I[:na1,:na2,:na3,:na4] = aaaa # aaaa
    I[na1:,na2:,na3:,na4:] = bbbb # bbbb
    I[:na1,na2:,:na3,na4:] = abab # abab
    I[:na1,na2:,na3:,:na4] = abba # abba
    I[na1:,:na2,:na3,na4:] = baab # baab
    I[na1:,:na2,na3:,:na4] = baba # baba
    return I

def int_to_spin(I_aaaa, I_bbbb, I_abab, noa, nva, nob, nvb):

    # vvvv
    aaaa = I_aaaa.vvvv - I_aaaa.vvvv.transpose((0,1,3,2))
    bbbb = I_bbbb.vvvv - I_bbbb.vvvv.transpose((0,1,3,2))
    abab = I_abab.vvvv
    baba = I_abab.vvvv.transpose((1,0,3,2))
    abba = -I_abab.vvvv.transpose((0,1,3,2))
    baab = -I_abab.vvvv.transpose((1,0,2,3))
    Ivvvv = int_to_spin_block(aaaa, bbbb, abab, baba, abba, baab,
            nva, nva, nva, nva, nvb, nvb, nvb, nvb)

    # vvvo
    aaaa = I_aaaa.vvvo - I_aaaa.vvov.transpose((0,1,3,2))
    bbbb = I_bbbb.vvvo - I_bbbb.vvov.transpose((0,1,3,2))
    abab = I_abab.vvvo
    baba = I_abab.vvov.transpose((1,0,3,2))
    abba = -I_abab.vvov.transpose((0,1,3,2))
    baab = -I_abab.vvvo.transpose((1,0,2,3))
    Ivvvo = int_to_spin_block(aaaa, bbbb, abab, baba, abba, baab,
            nva, nva, nva, noa, nvb, nvb, nvb, nob)

    # vovv
    aaaa = I_aaaa.vovv - I_aaaa.vovv.transpose((0,1,3,2))
    bbbb = I_bbbb.vovv - I_bbbb.vovv.transpose((0,1,3,2))
    abab = I_abab.vovv
    baba = I_abab.ovvv.transpose((1,0,3,2))
    abba = -I_abab.vovv.transpose((0,1,3,2))
    baab = -I_abab.ovvv.transpose((1,0,2,3))
    Ivovv = int_to_spin_block(aaaa, bbbb, abab, baba, abba, baab,
            nva, noa, nva, nva, nvb, nob, nvb, nvb)

    # vvoo
    aaaa = I_aaaa.vvoo - I_aaaa.vvoo.transpose((0,1,3,2))
    bbbb = I_bbbb.vvoo - I_bbbb.vvoo.transpose((0,1,3,2))
    abab = I_abab.vvoo
    baba = I_abab.vvoo.transpose((1,0,3,2))
    abba = -I_abab.vvoo.transpose((0,1,3,2))
    baab = -I_abab.vvoo.transpose((1,0,2,3))
    Ivvoo = int_to_spin_block(aaaa, bbbb, abab, baba, abba, baab,
            nva, nva, noa, noa, nvb, nvb, nob, nob)

    # vovo
    aaaa = I_aaaa.vovo - I_aaaa.voov.transpose((0,1,3,2))
    bbbb = I_bbbb.vovo - I_bbbb.voov.transpose((0,1,3,2))
    abab = I_abab.vovo
    baba = I_abab.ovov.transpose((1,0,3,2))
    abba = -I_abab.voov.transpose((0,1,3,2))
    baab = -I_abab.ovvo.transpose((1,0,2,3))
    Ivovo = int_to_spin_block(aaaa, bbbb, abab, baba, abba, baab,
            nva, noa, nva, noa, nvb, nob, nvb, nob)

    # oovv
    aaaa = I_aaaa.oovv - I_aaaa.oovv.transpose((0,1,3,2))
    bbbb = I_bbbb.oovv - I_bbbb.oovv.transpose((0,1,3,2))
    abab = I_abab.oovv
    baba = I_abab.oovv.transpose((1,0,3,2))
    abba = -I_abab.oovv.transpose((0,1,3,2))
    baab = -I_abab.oovv.transpose((1,0,2,3))
    Ioovv = int_to_spin_block(aaaa, bbbb, abab, baba, abba, baab,
            noa, noa, nva, nva, nob, nob, nvb, nvb)

    # vooo
    aaaa = I_aaaa.vooo - I_aaaa.vooo.transpose((0,1,3,2))
    bbbb = I_bbbb.vooo - I_bbbb.vooo.transpose((0,1,3,2))
    abab = I_abab.vooo
    baba = I_abab.ovoo.transpose((1,0,3,2))
    abba = -I_abab.vooo.transpose((0,1,3,2))
    baab = -I_abab.ovoo.transpose((1,0,2,3))
    Ivooo = int_to_spin_block(aaaa, bbbb, abab, baba, abba, baab,
            nva, noa, noa, noa, nvb, nob, nob, nob)

    # ooov
    aaaa = I_aaaa.ooov - I_aaaa.oovo.transpose((0,1,3,2))
    bbbb = I_bbbb.ooov - I_bbbb.oovo.transpose((0,1,3,2))
    abab = I_abab.ooov
    baba = I_abab.oovo.transpose((1,0,3,2))
    abba = -I_abab.oovo.transpose((0,1,3,2))
    baab = -I_abab.ooov.transpose((1,0,2,3))
    Iooov = int_to_spin_block(aaaa, bbbb, abab, baba, abba, baab,
            noa, noa, noa, nva, nob, nob, nob, nvb)

    # oooo
    aaaa = I_aaaa.oooo - I_aaaa.oooo.transpose((0,1,3,2))
    bbbb = I_bbbb.oooo - I_bbbb.oooo.transpose((0,1,3,2))
    abab = I_abab.oooo
    baba = I_abab.oooo.transpose((1,0,3,2))
    abba = -I_abab.oooo.transpose((0,1,3,2))
    baab = -I_abab.oooo.transpose((1,0,2,3))
    Ioooo = int_to_spin_block(aaaa, bbbb, abab, baba, abba, baab,
            noa, noa, noa, noa, nob, nob, nob, nob)

    return two_e_blocks(vvvv=Ivvvv,vvvo=Ivvvo,vovv=Ivovv,vvoo=Ivvoo,
            vovo=Ivovo,oovv=Ioovv,vooo=Ivooo,ooov=Iooov,oooo=Ioooo)

def int_to_spin2(I_aaaa, I_bbbb, I_abab, noa, nva, nob, nvb):

    # vvvv
    aaaa = I_aaaa.vvvv
    bbbb = I_bbbb.vvvv
    abab = I_abab.vvvv
    baba = I_abab.vvvv.transpose((1,0,3,2))
    abba = -I_abab.vvvv.transpose((0,1,3,2))
    baab = -I_abab.vvvv.transpose((1,0,2,3))
    Ivvvv = int_to_spin_block(aaaa, bbbb, abab, baba, abba, baab,
            nva, nva, nva, nva, nvb, nvb, nvb, nvb)

    # vvvo
    aaaa = I_aaaa.vvvo
    bbbb = I_bbbb.vvvo
    abab = I_abab.vvvo
    baba = I_abab.vvov.transpose((1,0,3,2))
    abba = -I_abab.vvov.transpose((0,1,3,2))
    baab = -I_abab.vvvo.transpose((1,0,2,3))
    Ivvvo = int_to_spin_block(aaaa, bbbb, abab, baba, abba, baab,
            nva, nva, nva, noa, nvb, nvb, nvb, nob)

    # vovv
    aaaa = I_aaaa.vovv
    bbbb = I_bbbb.vovv
    abab = I_abab.vovv
    baba = I_abab.ovvv.transpose((1,0,3,2))
    abba = -I_abab.vovv.transpose((0,1,3,2))
    baab = -I_abab.ovvv.transpose((1,0,2,3))
    Ivovv = int_to_spin_block(aaaa, bbbb, abab, baba, abba, baab,
            nva, noa, nva, nva, nvb, nob, nvb, nvb)

    # vvoo
    aaaa = I_aaaa.vvoo
    bbbb = I_bbbb.vvoo
    abab = I_abab.vvoo
    baba = I_abab.vvoo.transpose((1,0,3,2))
    abba = -I_abab.vvoo.transpose((0,1,3,2))
    baab = -I_abab.vvoo.transpose((1,0,2,3))
    Ivvoo = int_to_spin_block(aaaa, bbbb, abab, baba, abba, baab,
            nva, nva, noa, noa, nvb, nvb, nob, nob)

    # vovo
    aaaa = I_aaaa.vovo
    bbbb = I_bbbb.vovo
    abab = I_abab.vovo
    baba = I_abab.ovov.transpose((1,0,3,2))
    abba = -I_abab.voov.transpose((0,1,3,2))
    baab = -I_abab.ovvo.transpose((1,0,2,3))
    Ivovo = int_to_spin_block(aaaa, bbbb, abab, baba, abba, baab,
            nva, noa, nva, noa, nvb, nob, nvb, nob)

    # oovv
    aaaa = I_aaaa.oovv
    bbbb = I_bbbb.oovv
    abab = I_abab.oovv
    baba = I_abab.oovv.transpose((1,0,3,2))
    abba = -I_abab.oovv.transpose((0,1,3,2))
    baab = -I_abab.oovv.transpose((1,0,2,3))
    Ioovv = int_to_spin_block(aaaa, bbbb, abab, baba, abba, baab,
            noa, noa, nva, nva, nob, nob, nvb, nvb)

    # vooo
    aaaa = I_aaaa.vooo
    bbbb = I_bbbb.vooo
    abab = I_abab.vooo
    baba = I_abab.ovoo.transpose((1,0,3,2))
    abba = -I_abab.vooo.transpose((0,1,3,2))
    baab = -I_abab.ovoo.transpose((1,0,2,3))
    Ivooo = int_to_spin_block(aaaa, bbbb, abab, baba, abba, baab,
            nva, noa, noa, noa, nvb, nob, nob, nob)

    # ooov
    aaaa = I_aaaa.ooov
    bbbb = I_bbbb.ooov
    abab = I_abab.ooov
    baba = I_abab.oovo.transpose((1,0,3,2))
    abba = -I_abab.oovo.transpose((0,1,3,2))
    baab = -I_abab.ooov.transpose((1,0,2,3))
    Iooov = int_to_spin_block(aaaa, bbbb, abab, baba, abba, baab,
            noa, noa, noa, nva, nob, nob, nob, nvb)

    # oooo
    aaaa = I_aaaa.oooo
    bbbb = I_bbbb.oooo
    abab = I_abab.oooo
    baba = I_abab.oooo.transpose((1,0,3,2))
    abba = -I_abab.oooo.transpose((0,1,3,2))
    baab = -I_abab.oooo.transpose((1,0,2,3))
    Ioooo = int_to_spin_block(aaaa, bbbb, abab, baba, abba, baab,
            noa, noa, noa, noa, nob, nob, nob, nob)

    return two_e_blocks(vvvv=Ivvvv,vvvo=Ivvvo,vovv=Ivovv,vvoo=Ivvoo,
            vovo=Ivovo,oovv=Ioovv,vooo=Ivooo,ooov=Iooov,oooo=Ioooo)

def int_to_spatial(I, noa, nob, nva, nvb):
    vvvv = I.vvvv[:nva,:nva,:nva,:nva]
    vvvo = I.vvvo[:nva,:nva,:nva,:noa]
    vovv = I.vovv[:nva,:noa,:nva,:nva]
    vvoo = I.vvoo[:nva,:nva,:noa,:noa]
    vovo = I.vovo[:nva,:noa,:nva,:noa]
    oovv = I.oovv[:noa,:noa,:nva,:nva]
    vooo = I.vooo[:nva,:noa,:noa,:noa]
    ooov = I.ooov[:noa,:noa,:noa,:nva]
    oooo = I.oooo[:noa,:noa,:noa,:noa]
    Ia = two_e_blocks(vvvv=vvvv,
            vvvo=vvvo,vovv=vovv,vvoo=vvoo,vovo=vovo,
            oovv=oovv,vooo=vooo,ooov=ooov,oooo=oooo)

    vvvv = I.vvvv[nva:,nva:,nva:,nva:]
    vvvo = I.vvvo[nva:,nva:,nva:,noa:]
    vovv = I.vovv[nva:,noa:,nva:,nva:]
    vvoo = I.vvoo[nva:,nva:,noa:,noa:]
    vovo = I.vovo[nva:,noa:,nva:,noa:]
    oovv = I.oovv[noa:,noa:,nva:,nva:]
    vooo = I.vooo[nva:,noa:,noa:,noa:]
    ooov = I.ooov[noa:,noa:,noa:,nva:]
    oooo = I.oooo[noa:,noa:,noa:,noa:]
    Ib = two_e_blocks(vvvv=vvvv,
            vvvo=vvvo,vovv=vovv,vvoo=vvoo,vovo=vovo,
            oovv=oovv,vooo=vooo,ooov=ooov,oooo=oooo)

    vvvv = I.vvvv[:nva,nva:,:nva,nva:]
    vvvo = I.vvvo[:nva,nva:,:nva,noa:]
    vvov = -I.vvvo[:nva,nva:,nva:,:noa].transpose((0,1,3,2))
    vovv = I.vovv[:nva,noa:,:nva,nva:]
    ovvv = -I.vovv[nva:,:noa,:nva,nva:].transpose((1,0,2,3))
    vvoo = I.vvoo[:nva,nva:,:noa,noa:]
    vovo = I.vovo[:nva,noa:,:nva,noa:]
    voov = -I.vovo[:nva,noa:,nva:,:noa].transpose((0,1,3,2))
    ovov = I.vovo[nva:,:noa,nva:,:noa].transpose((1,0,3,2))
    ovvo = -I.vovo[nva:,:noa,:nva,noa:].transpose((1,0,2,3))
    oovv = I.oovv[:noa,noa:,:nva,nva:]
    vooo = I.vooo[:nva,noa:,:noa,noa:]
    ovoo = -I.vooo[nva:,:noa,:noa,noa:].transpose((1,0,2,3))
    ooov = I.ooov[:noa,noa:,:noa,nva:]
    oovo = -I.ooov[:noa,noa:,noa:,:nva].transpose((0,1,3,2))
    oooo = I.oooo[:noa,noa:,:noa,noa:]
    Iabab = two_e_blocks_full(vvvv=vvvv,
            vvvo=vvvo,vvov=vvov,
            vovv=vovv,ovvv=ovvv,
            vvoo=vvoo,vovo=vovo,
            ovvo=ovvo,voov=voov,
            ovov=ovov,oovv=oovv,
            vooo=vooo,ovoo=ovoo,
            oovo=oovo,ooov=ooov,
            oooo=oooo)

    return Ia,Ib,Iabab

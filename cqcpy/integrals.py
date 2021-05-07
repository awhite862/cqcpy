import numpy
import pyscf.ao2mo as mol_ao2mo


def get_chem(mol, o1, o2, o3, o4, anti=False):
    """Get ERIs in chemist's notation for given orbital Coeffs."""
    n1 = o1.shape[1]
    n2 = o2.shape[1]
    n3 = o3.shape[1]
    n4 = o4.shape[1]
    Id = mol_ao2mo.general(mol, (o1,o2,o3,o4), compact=False).reshape(n1, n2, n3, n4)
    if anti:
        Ix = mol_ao2mo.general(mol, (o1,o4,o3,o2), compact=False).reshape(n1, n4, n3, n2)
        return Id - Ix.transpose(0, 3, 2, 1)
    else:
        return Id


def get_chem_anti(mol, o1, o2, o3, o4):
    """Get antisymmetrized ERIs in chemist's notation for given orbital Coeffs."""
    return get_chem(mol, o1, o1, o3, o4, anti=True)


def get_phys(mol, o1, o2, o3, o4, anti=False):
    """Get ERIs in physicist's notation for given orbital Coeffs."""
    return get_chem(mol, o1, o3, o2, o4, anti=anti).transpose((0,2,1,3))


def get_phys_anti(mol, o1, o2, o3, o4):
    """Get antisymmetrized ERIs in physicist's notation for given orbital Coeffs."""
    return get_phys(mol, o1, o2, o3, o4, anti=True)


def get_chemu(mol, o1, o2, o3, o4, p1, p2, p3, p4, anti=False):
    """Get unrestricted, ERIs in chemist's
    notation for given alphd and beta orbital Coeffs.
    """
    dtype = o1.dtype
    Ia = get_chem(mol, o1, o2, o3, o4, anti=anti)
    Ib = get_chem(mol, p1, p2, p3, p4, anti=anti)
    Iaabb = get_chem(mol, o1, o2, p3, p4)
    Ibbaa = get_chem(mol, p1, p2, o3, o4)
    a1 = o1.shape[1]
    a2 = o2.shape[1]
    a3 = o3.shape[1]
    a4 = o4.shape[1]
    b1 = p1.shape[1]
    b2 = p2.shape[1]
    b3 = p3.shape[1]
    b4 = p4.shape[1]
    n1 = a1 + b1
    n2 = a2 + b2
    n3 = a3 + b3
    n4 = a4 + b4
    I = numpy.zeros((n1,n2,n3,n4), dtype=dtype)
    I[:a1,:a2,:a3,:a4] = Ia
    I[a1:,a2:,a3:,a4:] = Ib
    I[:a1,:a2,a3:,a4:] = Iaabb
    I[a1:,a2:,:a3,:a4] = Ibbaa
    if anti:
        Iaabbx = get_chem(mol, o1, o4, p3, p2)
        Ibbaax = get_chem(mol, p1, p4, o3, o2)
        I[:a1,a2:,a3:,:a4] = -Iaabbx.transpose((0,3,2,1))
        I[a1:,:a2,:a3,a4:] = -Ibbaax.transpose((0,3,2,1))
    return I


def get_chem_antiu(mol, o1, o2, o3, o4, p1, p2, p3, p4):
    """Get unrestricted, ERIs in chemist's
    notation for given alphd and beta orbital Coeffs.
    """
    return get_chemu(mol, o1, o2, o3, o4, p1, p2, p3, p4, anti=True)


def get_chemu_all(mol, oa, ob, anti=False):
    """Get unrestricted ERIs in chemist's
    notation for given orbital Coeffs.
    """
    na = oa.shape[1]
    nb = ob.shape[1]
    n = na + nb
    Ia = mol_ao2mo.general(
        mol, (oa,oa,oa,oa), compact=False).reshape(na, na, na, na)
    Ib = mol_ao2mo.general(
        mol, (ob,ob,ob,ob), compact=False).reshape(nb, nb, nb, nb)
    Iab = mol_ao2mo.general(
        mol, (oa,oa,ob,ob), compact=False).reshape(na, na, nb, nb)
    dtype = oa.dtype
    Id = numpy.zeros((n,n,n,n), dtype=dtype)
    Id[:na,:na,:na,:na] = Ia
    Id[na:,na:,na:,na:] = Ib
    Id[:na,:na,na:,na:] = Iab
    Id[na:,na:,:na,:na] = Iab.transpose((2,3,0,1))
    if anti:
        return Id - Id.transpose(0, 3, 2, 1)
    else:
        return Id


def get_chem_antiu_all(mol, oa, ob):
    """Get unrestricted, antisymmetrized ERIs in chemist's
    notation for given orbital Coeffs.
    """
    return get_chemu_all(mol, oa, ob, anti=True)


def get_physu_all(mol, oa, ob, anti=False):
    """Get unrestricted ERIs in physicist's
    notation for given orbital Coeffs.
    """
    I = get_chemu_all(mol, oa, ob, anti=anti)
    return I.transpose(0, 2, 1, 3)


def get_phys_antiu_all(mol, oa, ob):
    """Get unrestricted, antisymmetrized ERIs in physicist's
    notation for given orbital Coeffs.
    """
    return get_physu_all(mol, oa, ob, anti=True)


def get_phys_antiu(mol, o1, o2, o3, o4, p1, p2, p3, p4):
    """Get unrestricted, antisymmetrized ERIs in physicist's
    notation for given orbital Coeffs.
    """
    I = get_chem_antiu(mol, o1, o3, o2, o4, p1, p3, p2, p4)
    return I.transpose(0, 2, 1, 3)


def get_chem_sol(mf, o1, o2, o3, o4, anti=False):
    """Get ERIs in chemist's notation for given orbital Coeffs."""
    n1 = o1.shape[1]
    n2 = o2.shape[1]
    n3 = o3.shape[1]
    n4 = o4.shape[1]
    Id = mf.with_df.ao2mo((o1,o2,o3,o4), mf.kpt, compact=False).reshape(n1, n2, n3, n4)
    if anti:
        Ix = mf.with_df.ao2mo((o1,o4,o3,o2), mf.kpt, compact=False).reshape(n1, n4, n3, n2)
        return Id - Ix.transpose(0, 3, 2, 1)
    else:
        return Id


def get_phys_sol(mf, o1, o2, o3, o4, anti=False):
    return get_chem_sol(mf, o1, o3, o2, o4, anti=anti).transpose((0,2,1,3))


def get_chemu_sol(mf, o1, o2, o3, o4, p1, p2, p3, p4, anti=False):
    """Get unrestricted ERIs in chemist's
    notation for given orbital Coeffs.
    """
    Ia = get_chem_sol(mf, o1, o2, o3, o4, anti=anti)
    Ib = get_chem_sol(mf, p1, p2, p3, p4, anti=anti)
    Iaabb = get_chem_sol(mf, o1, o2, p3, p4)
    Ibbaa = get_chem_sol(mf, p1, p2, o3, o4)
    a1 = o1.shape[1]
    a2 = o2.shape[1]
    a3 = o3.shape[1]
    a4 = o4.shape[1]
    b1 = p1.shape[1]
    b2 = p2.shape[1]
    b3 = p3.shape[1]
    b4 = p4.shape[1]
    n1 = a1 + b1
    n2 = a2 + b2
    n3 = a3 + b3
    n4 = a4 + b4
    dtype = o1.dtype
    I = numpy.zeros((n1,n2,n3,n4), dtype=dtype)
    I[:a1,:a2,:a3,:a4] = Ia
    I[a1:,a2:,a3:,a4:] = Ib
    I[:a1,:a2,a3:,a4:] = Iaabb
    I[a1:,a2:,:a3,:a4] = Ibbaa
    if anti:
        Iaabbx = get_chem_sol(mf, o1, o4, p3, p2)
        Ibbaax = get_chem_sol(mf, p1, p4, o3, o2)
        I[:a1,a2:,a3:,:a4] = -Iaabbx.transpose((0,3,2,1))
        I[a1:,:a2,:a3,a4:] = -Ibbaax.transpose((0,3,2,1))
    return I


def get_chem_antiu_sol(mf, o1, o2, o3, o4, p1, p2, p3, p4):
    return get_chemu_sol(mf, o1, o2, o3, o4, p1, p2, p3, p4, anti=True)


def get_phys_antiu_sol(mf, o1, o2, o3, o4, p1, p2, p3, p4):
    """Get unrestricted, antisymmetrized ERIs in physicist's
    notation for given orbital Coeffs.
    """
    I = get_chem_antiu_sol(mf, o1, o3, o2, o4, p1, p3, p2, p4)
    return I.transpose(0, 2, 1, 3)


def get_chemu_all_sol(mf, oa, ob, anti=False):
    """Get unrestricted ERIs in chemist's
    notation for given orbital Coeffs.
    """
    na = oa.shape[1]
    nb = ob.shape[1]
    n = na + nb
    Ia = mf.with_df.ao2mo(
        (oa,oa,oa,oa), mf.kpt, compact=False).reshape(na, na, na, na)
    Ib = mf.with_df.ao2mo(
        (ob,ob,ob,ob), mf.kpt, compact=False).reshape(nb, nb, nb, nb)
    Iab = mf.with_df.ao2mo(
        (oa,oa,ob,ob), mf.kpt, compact=False).reshape(na, na, nb, nb)
    Id = numpy.zeros((n,n,n,n), dtype=oa.dtype)
    Id[:na,:na,:na,:na] = Ia
    Id[na:,na:,na:,na:] = Ib
    Id[:na,:na,na:,na:] = Iab
    Id[na:,na:,:na,:na] = Iab.transpose((2,3,0,1))
    if anti:
        return Id - Id.transpose(0, 3, 2, 1)
    else:
        return Id


def get_physu_all_sol(mf, oa, ob, anti=False):
    return get_chemu_all_sol(mf, oa, ob, anti=anti).transpose((0,2,1,3))


def get_physu_all_gen(mf, anti=False):
    """Get unrestricted, ERIs in physicist's
    notation for given scf reference.
    """
    pbc = hasattr(mf, "kpt")

    if pbc:
        if len(mf.mo_occ.shape) == 1:
            mo = mf.mo_coeff
            return get_physu_all_sol(mf, mo, mo, anti=anti)
        elif len(mf.mo_occ.shape) == 2:
            moa = mf.mo_coeff[0]
            mob = mf.mo_coeff[1]
            return get_physu_all_sol(mf, moa, mob, anti=anti)
    else:
        mol = mf.mol
        if len(mf.mo_occ.shape) == 1:
            mo = mf.mo_coeff
            return get_physu_all(mol, mo, mo, anti=anti)
        elif len(mf.mo_occ.shape) == 2:
            moa = mf.mo_coeff[0]
            mob = mf.mo_coeff[1]
            return get_physu_all(mol, moa, mob, anti=anti)


def get_phys_antiu_all_gen(mf):
    """Get spin-orbital antisymmetrized ERIs in physicist's
    notation for given scf reference.
    """
    return get_physu_all_gen(mf, anti=True)


def get_phys_gen(mf, mo1, mo2, mo3, mo4, anti=False):
    pbc = hasattr(mf, "kpt")

    if pbc:
        return get_phys_sol(mf, mo1, mo2, mo3, mo4, anti=anti)
    else:
        return get_phys(mf.mol, mo1, mo2, mo3, mo4, anti=anti)


class eri_blocks(object):
    """Object for storing ERI blocks.

    Atributes:
        has_xxxx(bool): Does the object store the xxxx block
        xxxx (array): The xxxx block of integrals

    xxxx = vvvv, vvvo, vovv, vvoo, vovo, oovv, vooo, ooov, oooo
    """
    def __init__(self, mf, code=0):
        self.has_vvvv = False
        self.has_vvvo = False
        self.has_vovv = False
        self.has_vvoo = False
        self.has_vovo = False
        self.has_oovv = False
        self.has_vooo = False
        self.has_ooov = False
        self.has_oooo = False
        if code == 1 or code == 0:
            self.has_vvvv = True
        if code == 2 or code == 0:
            self.has_vvvo = True
        if code == 3 or code == 0:
            self.has_vovv = True
        if code == 4 or code == 0:
            self.has_vvoo = True
        if code == 5 or code == 0:
            self.has_vovo = True
        if code == 6 or code == 0:
            self.has_oovv = True
        if code == 7 or code == 0:
            self.has_vooo = True
        if code == 8 or code == 0:
            self.has_ooov = True
        if code == 9 or code == 0:
            self.has_oooo = True
        mo_occ = mf.mo_occ
        pbc = hasattr(mf, "kpt")

        if len(mo_occ.shape) == 1:
            o = mf.mo_coeff[:,mo_occ > 0]
            v = mf.mo_coeff[:,mo_occ == 0]
            if pbc:
                self._build_integrals_sol(mf, o, o, v, v)
            else:
                self._build_integrals_mol(mf, o, o, v, v)
        elif len(mo_occ.shape) == 2:
            mo_occa = mo_occ[0]
            mo_occb = mo_occ[1]
            oa = (mf.mo_coeff[0])[:,mo_occa > 0]
            va = (mf.mo_coeff[0])[:,mo_occa == 0]
            ob = (mf.mo_coeff[1])[:,mo_occb > 0]
            vb = (mf.mo_coeff[1])[:,mo_occb == 0]
            if pbc:
                self._build_integrals_sol(mf, oa, ob, va, vb)
            else:
                self._build_integrals_mol(mf, oa, ob, va, vb)

    def _build_integrals_mol(self, mf, oa, ob, va, vb):
        if self.has_vvvv:
            self.vvvv = get_phys_antiu(mf.mol, va, va, va, va, vb, vb, vb, vb)
        if self.has_vvvo:
            self.vvvo = get_phys_antiu(mf.mol, va, va, va, oa, vb, vb, vb, ob)
        if self.has_vovv:
            self.vovv = get_phys_antiu(mf.mol, va, oa, va, va, vb, ob, vb, vb)
        if self.has_vvoo:
            self.vvoo = get_phys_antiu(mf.mol, va, va, oa, oa, vb, vb, ob, ob)
        if self.has_vovo:
            self.vovo = get_phys_antiu(mf.mol, va, oa, va, oa, vb, ob, vb, ob)
        if self.has_oovv:
            self.oovv = get_phys_antiu(mf.mol, oa, oa, va, va, ob, ob, vb, vb)
        if self.has_vooo:
            self.vooo = get_phys_antiu(mf.mol, va, oa, oa, oa, vb, ob, ob, ob)
        if self.has_ooov:
            self.ooov = get_phys_antiu(mf.mol, oa, oa, oa, va, ob, ob, ob, vb)
        if self.has_oooo:
            self.oooo = get_phys_antiu(mf.mol, oa, oa, oa, oa, ob, ob, ob, ob)

    def _build_integrals_sol(self, mf, oa, ob, va, vb):
        if self.has_vvvv:
            self.vvvv = get_phys_antiu_sol(mf, va, va, va, va, vb, vb, vb, vb)
        if self.has_vvvo:
            self.vvvo = get_phys_antiu_sol(mf, va, va, va, oa, vb, vb, vb, ob)
        if self.has_vovv:
            self.vovv = get_phys_antiu_sol(mf, va, oa, va, va, vb, ob, vb, vb)
        if self.has_vvoo:
            self.vvoo = get_phys_antiu_sol(mf, va, va, oa, oa, vb, vb, ob, ob)
        if self.has_vovo:
            self.vovo = get_phys_antiu_sol(mf, va, oa, va, oa, vb, ob, vb, ob)
        if self.has_oovv:
            self.oovv = get_phys_antiu_sol(mf, oa, oa, va, va, ob, ob, vb, vb)
        if self.has_vooo:
            self.vooo = get_phys_antiu_sol(mf, va, oa, oa, oa, vb, ob, ob, ob)
        if self.has_ooov:
            self.ooov = get_phys_antiu_sol(mf, oa, oa, oa, va, ob, ob, ob, vb)
        if self.has_oooo:
            self.oooo = get_phys_antiu_sol(mf, oa, oa, oa, oa, ob, ob, ob, ob)

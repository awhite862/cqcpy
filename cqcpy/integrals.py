import numpy
from pyscf import gto, scf
import pyscf.ao2mo as mol_ao2mo
#import pyscf.pbc.df.df_ao2mo as sol_ao2mo

def get_chem(mol,o1,o2,o3,o4):
    """Get ERIs in chemist's notation for given orbital Coeffs."""
    n1 = o1.shape[1]
    n2 = o2.shape[1]
    n3 = o3.shape[1]
    n4 = o4.shape[1]
    I = mol_ao2mo.general(mol,(o1,o2,o3,o4),compact=False).reshape(n1,n2,n3,n4)
    return I

def get_chem_anti(mol,o1,o2,o3,o4):
    """Get antisymmetrized ERIs in chemist's notation for given orbital Coeffs."""
    n1 = o1.shape[1]
    n2 = o2.shape[1]
    n3 = o3.shape[1]
    n4 = o4.shape[1]
    Id = mol_ao2mo.general(mol,(o1,o2,o3,o4),compact=False).reshape(n1,n2,n3,n4)
    Ix = mol_ao2mo.general(mol,(o1,o4,o3,o2),compact=False).reshape(n1,n4,n3,n2)
    return Id - Ix.transpose(0,3,2,1)
    

def get_phys(mol,o1,o2,o3,o4):
    """Get ERIs in physicist's notation for given orbital Coeffs."""
    n1 = o1.shape[1]
    n2 = o2.shape[1]
    n3 = o3.shape[1]
    n4 = o4.shape[1]
    I = mol_ao2mo.general(mol,(o1,o3,o2,o4),compact=False).reshape(n1,n3,n2,n4)
    I2 = I.transpose(0,2,1,3)
    return I2

def get_phys_anti(mol,o1,o2,o3,o4):
    """Get antisymmetrized ERIs in physicist's notation for given orbital Coeffs."""
    n1 = o1.shape[1]
    n2 = o2.shape[1]
    n3 = o3.shape[1]
    n4 = o4.shape[1]
    Id = mol_ao2mo.general(mol,(o1,o3,o2,o4),compact=False).reshape(n1,n3,n2,n4)
    Ix = mol_ao2mo.general(mol,(o1,o4,o2,o3),compact=False).reshape(n1,n4,n2,n3)
    return Id.transpose(0,2,1,3) - Ix.transpose(0,2,3,1)

def get_chem_antiu(mol,o1,o2,o3,o4,p1,p2,p3,p4,anti=True):
    """Get unrestricted, antisymmetrized ERIs in chemist's
    notation for given orbital Coeffs.
    """
    n1 = o1.shape[1]
    n2 = o2.shape[1]
    n3 = o3.shape[1]
    n4 = o4.shape[1]
    m1 = p1.shape[1]
    m2 = p2.shape[1]
    m3 = p3.shape[1]
    m4 = p4.shape[1]
    q1 = numpy.hstack((o1,p1))
    q2 = numpy.hstack((o2,p2))
    q3 = numpy.hstack((o3,p3))
    q4 = numpy.hstack((o4,p4))
    Id = mol_ao2mo.general(mol,(q1,q2,q3,q4),compact=False).reshape(
        n1+m1,n2+m2,n3+m3,n4+m4)
    Ix = mol_ao2mo.general(mol,(q1,q4,q3,q2),compact=False).reshape(
        n1+m1,n4+m4,n3+m3,n2+m2)
    Id[:n1,n2:,:,:] = 0
    Id[n1:,:n2,:,:] = 0
    Id[:,:,:n3,n4:] = 0
    Id[:,:,n3:,:n4] = 0
    Ix[:n1,n4:,:,:] = 0
    Ix[n1:,:n4,:,:] = 0
    Ix[:,:,:n3,n2:] = 0
    Ix[:,:,n3:,:n2] = 0
    if anti:
        return Id - Ix.transpose(0,3,2,1)
    else:
        return Id

def get_chem_antiu_all(mol,oa,ob,anti=True):
    """Get unrestricted, antisymmetrized ERIs in chemist's
    notation for given orbital Coeffs.
    """
    na = oa.shape[1]
    nb = ob.shape[1]
    n = na + nb
    ofull = numpy.hstack((oa,ob))
    Id = mol_ao2mo.general(mol,(ofull,ofull,ofull,ofull),
        compact=False).reshape(n,n,n,n)
    Id[:na,na:,:,:] = 0
    Id[na:,:na,:,:] = 0
    Id[:,:,:na,na:] = 0
    Id[:,:,na:,:na] = 0
    if anti:
        return Id - Id.transpose(0,3,2,1)
    else:
        return Id

def get_phys_antiu_all(mol,oa,ob,anti=True):
    """Get unrestricted, antisymmetrized ERIs in physicist's
    notation for given orbital Coeffs.
    """
    I = get_chem_antiu_all(mol,oa,ob,anti=anti)
    return I.transpose(0,2,1,3)

def get_phys_antiu(mol,o1,o2,o3,o4,p1,p2,p3,p4):
    """Get unrestricted, antisymmetrized ERIs in physicist's
    notation for given orbital Coeffs.
    """
    I = get_chem_antiu(mol,o1,o3,o2,o4,p1,p3,p2,p4)
    return I.transpose(0,2,1,3)

def get_phys_antiu_all_gen(mf,anti=True):
    """Get unrestricted, antisymmetrized ERIs in physicist's
    notation for given scf reference.
    """
    mol = mf.mol
    if len(mf.mo_occ.shape) == 1:
        mo = mf.mo_coeff
        return get_phys_antiu_all(mol,mo,mo,anti=anti)
    elif len(mf.mo_occ.shape) == 2:
        moa = mf.mo_coeff[0]
        mob = mf.mo_coeff[1]
        return get_phys_antiu_all(mol,moa,mob,anti=anti)

def get_chem_antiu_sol(mf,o1,o2,o3,o4,p1,p2,p3,p4,anti=True):
    """Get unrestricted, antisymmetrized ERIs in chemist's
    notation for given orbital Coeffs.
    """
    n1 = o1.shape[1]
    n2 = o2.shape[1]
    n3 = o3.shape[1]
    n4 = o4.shape[1]
    m1 = p1.shape[1]
    m2 = p2.shape[1]
    m3 = p3.shape[1]
    m4 = p4.shape[1]
    q1 = numpy.hstack((o1,p1))
    q2 = numpy.hstack((o2,p2))
    q3 = numpy.hstack((o3,p3))
    q4 = numpy.hstack((o4,p4))
    Id = mf.with_df.ao2mo((q1,q2,q3,q4),None,compact=False).reshape(
        n1+m1,n2+m2,n3+m3,n4+m4)
    Ix = mf.with_df.ao2mo((q1,q4,q3,q2),None,compact=False).reshape(
        n1+m1,n4+m4,n3+m3,n2+m2)
    Id[:n1,n2:,:,:] = 0
    Id[n1:,:n2,:,:] = 0
    Id[:,:,:n3,n4:] = 0
    Id[:,:,n3:,:n4] = 0
    Ix[:n1,n4:,:,:] = 0
    Ix[n1:,:n4,:,:] = 0
    Ix[:,:,:n3,n2:] = 0
    Ix[:,:,n3:,:n2] = 0
    if anti:
        return Id - Ix.transpose(0,3,2,1)
    else:
        return Id

def get_phys_antiu_sol(mf,o1,o2,o3,o4,p1,p2,p3,p4):
    """Get unrestricted, antisymmetrized ERIs in physicist's
    notation for given orbital Coeffs.
    """
    I = get_chem_antiu_sol(mf,o1,o3,o2,o4,p1,p3,p2,p4)
    return I.transpose(0,2,1,3)

# TODO: clean this up
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
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        print("Integrals object ......")
        if len(mo_occ.shape) == 1:
            print("RHF")
            o = mf.mo_coeff[:,mo_occ>0]
            v = mf.mo_coeff[:,mo_occ==0]
            try:
                ktemp = mf.kpt
                print("Solid state code")
                self._build_integrals_sol(mf,o,o,v,v)
            except AttributeError:
                self._build_integrals_mol(mf,o,o,v,v)
        elif len(mo_occ.shape) == 2:
            print("UHF")
            mo_occa = mo_occ[0]
            mo_occb = mo_occ[1]
            oa = (mf.mo_coeff[0])[:,mo_occa>0]
            va = (mf.mo_coeff[0])[:,mo_occa==0]
            ob = (mf.mo_coeff[1])[:,mo_occb>0]
            vb = (mf.mo_coeff[1])[:,mo_occb==0]
            try:
                ktemp = mf.kpt
                self._build_integrals_sol(mf,oa,ob,va,vb)
            except AttributeError:
                self._build_integrals_mol(mf,oa,ob,va,vb)

    def _build_integrals_mol(self, mf, oa, ob, va, vb):
        if self.has_vvvv:
            self.vvvv = get_phys_antiu(mf.mol,va,va,va,va,vb,vb,vb,vb)
        if self.has_vvvo:
            self.vvvo = get_phys_antiu(mf.mol,va,va,va,oa,vb,vb,vb,ob)
        if self.has_vovv:
            self.vovv = get_phys_antiu(mf.mol,va,oa,va,va,vb,ob,vb,vb)
        if self.has_vvoo:
            self.vvoo = get_phys_antiu(mf.mol,va,va,oa,oa,vb,vb,ob,ob)
        if self.has_vovo:
            self.vovo = get_phys_antiu(mf.mol,va,oa,va,oa,vb,ob,vb,ob)
        if self.has_oovv:
            self.oovv = get_phys_antiu(mf.mol,oa,oa,va,va,ob,ob,vb,vb)
        if self.has_vooo:
            self.vooo = get_phys_antiu(mf.mol,va,oa,oa,oa,vb,ob,ob,ob)
        if self.has_ooov:
            self.ooov = get_phys_antiu(mf.mol,oa,oa,oa,va,ob,ob,ob,vb)
        if self.has_oooo:
            self.oooo = get_phys_antiu(mf.mol,oa,oa,oa,oa,ob,ob,ob,ob)

    def _build_integrals_sol(self, mf, oa, ob, va, vb):
        if self.has_vvvv:
            self.vvvv = get_phys_antiu_sol(mf.mol,va,va,va,va,vb,vb,vb,vb)
        if self.has_vvvo:
            self.vvvo = get_phys_antiu_sol(mf.mol,va,va,va,oa,vb,vb,vb,ob)
        if self.has_vovv:
            self.vovv = get_phys_antiu_sol(mf.mol,va,oa,va,va,vb,ob,vb,vb)
        if self.has_vvoo:
            self.vvoo = get_phys_antiu_sol(mf.mol,va,va,oa,oa,vb,vb,ob,ob)
        if self.has_vovo:
            self.vovo = get_phys_antiu_sol(mf.mol,va,oa,va,oa,vb,ob,vb,ob)
        if self.has_oovv:
            self.oovv = get_phys_antiu_sol(mf.mol,oa,oa,va,va,ob,ob,vb,vb)
        if self.has_vooo:
            self.vooo = get_phys_antiu_sol(mf.mol,va,oa,oa,oa,vb,ob,ob,ob)
        if self.has_ooov:
            self.ooov = get_phys_antiu_sol(mf.mol,oa,oa,oa,va,ob,ob,ob,vb)
        if self.has_oooo:
            self.oooo = get_phys_antiu_sol(mf.mol,oa,oa,oa,oa,ob,ob,ob,ob)

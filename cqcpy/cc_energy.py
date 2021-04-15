import numpy
try:
    from pyscf import lib
    einsum = lib.einsum
except:
    einsum = numpy.einsum

def cc_energy_d(t2, eri):
    """Return the T2 contribution to the CC energy."""
    Ed = 0.25*einsum('abij,ijab->', t2, eri)
    return Ed

def cc_energy_s1(t1, f):
    """Return the T1 contribution to the CC energy."""
    Es1 = einsum('ai,ia->', t1, f)
    return Es1

def cc_energy_s2(t1, eri):
    """Return the T1**2 contribution to the CC energy."""
    Es2 = einsum('ai,bj,ijab->', t1, t1, eri)
    return Es2

def cc_energy_s2_a(t11, t12, eri):
    Es2 = einsum('ai,bj,ijab->', t11, t12, eri)
    return Es2

def cc_energy(t1, t2, f, eri):
    """Return the coupled cluster energy."""
    Ed = cc_energy_d(t2,eri)
    Es1 = cc_energy_s1(t1,f)
    Es2 = cc_energy_s2(t1,eri)
    return Ed + Es1 + 0.5*Es2

def mp2_energy(t1, t2, f, eri):
    """Return the coupled cluster energy."""
    Ed = cc_energy_d(t2,eri)
    Es1 = cc_energy_s1(t1,f)
    return Ed + Es1

def ucc_energy(t1, t2, fa, fb, Ia, Ib, Iabab):
    t1a,t1b = t1
    t2aa,t2ab,t2bb = t2
    tauaa = t2aa + 2.0*einsum('ai,bj->abij',t1a,t1a)
    taubb = t2bb + 2.0*einsum('ai,bj->abij',t1b,t1b)
    tauab = t2ab + einsum('ai,bj->abij',t1a,t1b)
    s = einsum('ia,ai->',fa,t1a)
    s += einsum('ia,ai->',fb,t1b)
    d = 0.25*einsum('ijab,abij->',Ia,tauaa)
    d += 0.25*einsum('ijab,abij->',Ib,taubb)
    d += einsum('ijab,abij->',Iabab,tauab)
    return d + s

def rcc_energy(t1, t2, f, I):
    s = 2.0*einsum('ia,ai->',f,t1)
    tau = t2 + einsum('ai,bj->abij',t1,t1)
    d = 2*einsum('abij,ijab->',tau,I)
    d -= einsum('abij,ijba->',tau,I)
    return d + s

def ump2_energy(t1, t2, fa, fb, Ia, Ib, Iabab):
    t1a,t1b = t1
    t2aa,t2ab,t2bb = t2
    s = einsum('ia,ai->',fa,t1a)
    s += einsum('ia,ai->',fb,t1b)
    d = 0.25*einsum('ijab,abij->',Ia,t2aa)
    d += 0.25*einsum('ijab,abij->',Ib,t2bb)
    d += einsum('ijab,abij->',Iabab,t2ab)
    return d + s

import numpy as np
from pyscf.pbc import scf, gto, mp
from pyscf.pbc.lib import kpts_helper
from cqcpy.integrals import get_phys_solk

cell = gto.Cell()
cell.atom='''
C 0.000000000000   0.000000000000   0.000000000000
C 1.685068664391   1.685068664391   1.685068664391
'''
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.unit = 'B'
cell.verbose = 4
cell.build()

kpts = cell.make_kpts([2, 2, 2])
nkpts = len(kpts)
kmf = scf.KRHF(cell, kpts, exxdiv=None)
kmf.conv_tol = 1e-12
kmf.conv_tol_grad = 1e-8
Ehf = kmf.kernel()
print("KHF energy (per unit cell) =", Ehf)
nmo = len(kmf.mo_occ[0])

# Here we assume that the number of occupied orbitals
# is the same at each k-point
nocc = np.count_nonzero(kmf.mo_occ[0])
nvir = nmo - nocc

mypt = mp.KMP2(kmf)
mypt.kernel()
print("KMP2 energy (per unit cell) =", mypt.e_tot)

# compute ERIs (physicists notation)
kpt_eris = np.zeros((nkpts, nkpts, nkpts, nmo, nmo, nmo, nmo), dtype=complex)
khelper = kpts_helper.KptsHelper(kmf.cell, kmf.kpts)
kconserv = khelper.kconserv
for kp in range(nkpts):
    for kq in range(nkpts):
        for kr in range(nkpts):
            ks = kconserv[kp, kr, kq]
            mop = kmf.mo_coeff[kp]
            moq = kmf.mo_coeff[kq]
            mor = kmf.mo_coeff[kr]
            mos = kmf.mo_coeff[ks]
            kpt_eris[kp, kq, kr] = get_phys_solk(
                kmf, (kp, kq, kr, ks), mop, moq, mor, mos)/nkpts

# compute the MP2 T2
kpt_abij = np.zeros(
    (nkpts, nkpts, nkpts, nvir, nvir, nocc, nocc), dtype=complex)
for ka in range(nkpts):
    for kb in range(nkpts):
        for ki in range(nkpts):
            kj = kconserv[ka, ki, kb]
            D2 = kmf.mo_energy[ka][nocc:][:, None, None, None]\
                + kmf.mo_energy[kb][nocc:][None, :, None, None]\
                - kmf.mo_energy[ki][:nocc][None, None, :, None]\
                - kmf.mo_energy[kj][:nocc][None, None, None, :]
            kpt_abij[ka, kb, ki] = \
                    -1*kpt_eris[ka, kb, ki, nocc:, nocc:, :nocc, :nocc] / D2

# compute the KMP2 energy and compare
emp2 = 0
for ka in range(nkpts):
    for kb in range(nkpts):
        for ki in range(nkpts):
            kj = kconserv[ka, ki, kb]
            I2 = kpt_eris[ki, kj, ka, :nocc, :nocc, nocc:, nocc:]
            I2x = kpt_eris[ki, kj, kb, :nocc, :nocc, nocc:, nocc:]
            emp2 += 2.0*np.einsum('ijab,abij->', I2, kpt_abij[ka, kb, ki]).real
            emp2 -= np.einsum('ijba,abij->', I2x, kpt_abij[ka, kb, ki]).real


print("KMP2 energy (out) =", emp2/nkpts)
print("KMP2 energy (ref) =", mypt.e_corr)

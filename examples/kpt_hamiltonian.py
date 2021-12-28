import numpy as np
from pyscf.pbc import scf, gto, mp
from cqcpy.integrals import get_solk_integrals

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

# compute the integrals
#h1, h2 = get_solk_integrals(kmf, eriorder='phys')
h1, h2 = get_solk_integrals(kmf, eriorder='chem')
h2 = h2/nkpts

# test by computing the SCF energy
Escf = kmf.energy_nuc()
for k in range(nkpts): 
    Escf += 2.0*np.einsum('ii->', h1[k, :nocc, :nocc]) / nkpts

for kp in range(nkpts):
    for kq in range(nkpts):
        #Escf += 2.0*np.einsum(
        #    'ijij->', h2[kp, kq, kp, :nocc, :nocc, :nocc, :nocc]) / nkpts
        Escf += 2.0*np.einsum(
            'iijj->', h2[kp, kp, kq, :nocc, :nocc, :nocc, :nocc]) / nkpts
        Escf -= np.einsum(
            'ijji->', h2[kp, kq, kq, :nocc, :nocc, :nocc, :nocc]) / nkpts

print(Escf)
print(Ehf)

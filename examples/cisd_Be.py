import numpy
from pyscf import gto, scf, ci

from cqcpy import ci_utils
from cqcpy import integrals

mol = gto.M(
    atom = 'Be 0 0 0',
    basis = '631g')
mf = scf.RHF(mol)
mf.conv_tol = 1e-12
Escf = mf.kernel()
print(Escf)
myci = ci.CISD(mf).run()
print('RCISD correlation energy', myci.e_corr)


# run naive CISD calculation
mos = mf.mo_coeff
nmo = mos.shape[1]
hcore = mf.get_hcore()
ha = numpy.einsum('mp,mn,nq->pq',mos,hcore,mos)
hb = ha.copy()
I = integrals.get_phys(mol, mos, mos, mos, mos)

N = mol.nelectron
N = mol.nelectron
na = N//2
nb = na
sa = ci_utils.s_strings(nmo, na)
sb = ci_utils.s_strings(nmo, nb)
da = ci_utils.d_strings(nmo, na)
db = ci_utils.d_strings(nmo, nb)
occ = [1 if i < na else 0 for i in range(nmo)]
ref = ci_utils.Dstring(nmo,occ)
basis = []

# scf ground state
basis.append((ref,ref))

# singly excited states
for a in sa:
    basis.append((a,ref))
for b in sb:
    basis.append((ref,b))

# doubly excited states
for a in  da:
    basis.append((a,ref))
for b in  db:
    basis.append((ref,b))
for a in sa:
    for b in sb:
        basis.append((a,b))

nd = len(basis)
H = numpy.zeros((nd,nd))
const = -Escf + mol.energy_nuc()
for i,b in enumerate(basis):
    for j,k in enumerate(basis):
        H[i,j] = ci_utils.ci_matrixel(b[0],b[1],k[0],k[1],ha,hb,I,I,I,const)

eout,v = numpy.linalg.eigh(H)
#print(eout[0])
print('RCISD correlation energy', eout[0])

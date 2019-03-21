import numpy
from pyscf import gto, scf, dft
from pyscf.tdscf import rhf

from cqcpy import ci_utils
from cqcpy import integrals

# reference
mol = gto.Mole()
mol.build(
    atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
    basis = '631g'
)

mf = scf.RHF(mol)
mf.verbose = 0
mf.conv_tol = 1e-12
mf.conv_tol_grad = 1e-8
Escf = mf.kernel()

tda = mf.TDA()
tda.conv_tol = 1e-12
tda.nstates = 12
erefs = tda.kernel()[0]
tda.singlet = False
ereft = tda.kernel()[0]

# run CIS calculation
mos = mf.mo_coeff
nmo = mos.shape[1]
hcore = mf.get_hcore()
ha = numpy.einsum('mp,mn,nq->pq',mos,hcore,mos)
hb = ha.copy()
I = integrals.get_phys(mol, mos, mos, mos, mos)

N = mol.nelectron
na = N//2
nb = na
sa = ci_utils.s_strings(nmo, na)
sb = ci_utils.s_strings(nmo, nb)
occ = [1 if i < na else 0 for i in range(nmo)]
ref = ci_utils.Dstring(nmo,occ)
basis = []
for a in sa:
    basis.append((a,ref))
for b in sb:
    basis.append((ref,b))

nd = len(sa)+len(sb)
H = numpy.zeros((nd,nd))
const = -Escf + mol.energy_nuc()
F = mf.get_fock()
F = numpy.einsum('mp,mn,nq->pq',mos,F,mos)
mo_energy = mf.mo_energy
for i,b in enumerate(basis):
    for j,k in enumerate(basis):
        H[i,j] = ci_utils.ci_matrixel(b[0],b[1],k[0],k[1],ha,hb,I,I,I,const)

eout,v = numpy.linalg.eigh(H)

eref = numpy.sort(numpy.concatenate((erefs,ereft)))
print(eref[0:12])
print(eout[0:12])
#print(eref[0:12] - eout[0:12])

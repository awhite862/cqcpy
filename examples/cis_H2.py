import numpy
from pyscf import gto
from pyscf.tdscf.rhf import scf

from cqcpy import ci_utils
from cqcpy import integrals

# reference
mol = gto.Mole()
mol.build(
    atom='H 0 0 0; H 0 0 0.77',  # in Angstrom
    basis='631g'
)

mf = scf.RHF(mol)
mf.conv_tol_grad = 1e-9
Escf = mf.kernel()
mf.verbose = 0
tda = mf.TDA()
tda.nstates = 6
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
basis = ci_utils.ucis_basis(nmo, na, nb, gs=False)
nd = len(basis)

H = numpy.zeros((nd,nd))
const = -Escf + mol.energy_nuc()
F = mf.get_fock()
F = numpy.einsum('mp,mn,nq->pq',mos,F,mos)
mo_energy = mf.mo_energy
for i,b in enumerate(basis):
    for j,k in enumerate(basis):
        H[i,j] = ci_utils.ci_matrixel(b[0],b[1],k[0],k[1],ha,hb,I,I,I,const)

eout,v = numpy.linalg.eigh(H)
ns = na*(nmo - na)
eref = numpy.sort(numpy.concatenate((erefs,ereft)))
print(eref)
print(eout)

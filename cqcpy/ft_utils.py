import numpy

hartree_to_ev = 27.2113860217
kb = 8.617330350e-5 # in eV

def HtoK(T):
    """Convert KbT in Ha to Kelvin."""
    return T*hartree_to_ev / kb

def fermi_function(beta, epsilon, mu):
    """Return the Fermi-Dirac distribution function."""
    #return 1.0 / (numpy.exp(beta*(epsilon - mu)) + 1.0)
    emm = epsilon - mu
    x = beta*emm
    if x < -30.0:
        return 1.0 - numpy.exp(x)
    elif x > 30.0:
        return numpy.exp(-x)
    else:
        return 1.0 / (numpy.exp(x) + 1.0)

def vfermi_function(beta, epsilon, mu):
    """Return the complement of Fermi-Dirac distribution function."""
    emm = epsilon - mu
    x = beta*emm
    if x > 30.0:
        return 1.0 - numpy.exp(-x)
    elif x < -30.0:
        return numpy.exp(x)
    else:
        return numpy.exp(x) / (numpy.exp(x) + 1.0)

def ffv(beta, epsilon, mu):
    """Return the complement of Fermi-Dirac distribution function
    for a vector of energies."""
    fvir = numpy.zeros(epsilon.shape)
    for i in range(epsilon.shape[0]):
        fvir[i] = vfermi_function(beta, epsilon[i], mu)

    return fvir

def ff(beta, epsilon, mu):
    """Return the Fermi-Dirac distribution function
    for a vector of energies."""
    focc = numpy.zeros(epsilon.shape)
    for i in range(epsilon.shape[0]):
        focc[i] = fermi_function(beta, epsilon[i], mu)

    return focc

def grand_potential0(beta, epsilon, mu):
    """Return the 0th order Grand potential."""
    emm = epsilon - mu
    x = beta*emm
    if x < -20.0:
        return ((-numpy.exp(x))/beta + emm)
    elif x > 20.0:
        #return 0.0
        return ((-numpy.exp(-x))/beta)
    else:
        return numpy.log(fermi_function(beta, epsilon, mu))/beta + emm

def dgrand_potential0(beta, epsilon, mu):
    """Return the derivative of the 0th order Grand potential."""
    emm = epsilon - mu
    x = beta*emm
    if x < -20.0:
        return -emm*vfermi_function(beta, epsilon, mu)/beta + numpy.exp(x)/(beta/beta)
    elif x > 20.0:
        return emm*fermi_function(beta, epsilon, mu)/beta + numpy.exp(-x)/(beta*beta)
    else:
        return -numpy.log(fermi_function(beta, epsilon, mu))/(beta*beta)\
            - emm*vfermi_function(beta, epsilon, mu)/beta

def GP0(beta, epsilon, mu):
    """Return a vector of the 0th order Grand potentials for a
    many-body system.
    """
    argA = numpy.zeros(epsilon.shape)
    for i in range(epsilon.shape[0]):
        argA[i] = grand_potential0(beta, epsilon[i], mu)

    return argA

def uGP0(beta, ea, eb, mu):
    """Return a vector of the 0th order Grand potentials for a
    many-body system.
    """
    argA = numpy.zeros(ea.shape)
    argB = numpy.zeros(eb.shape)
    for i in range(ea.shape[0]):
        argA[i] = grand_potential0(beta, ea[i], mu)
    for i in range(eb.shape[0]):
        argB[i] = grand_potential0(beta, eb[i], mu)

    return argA,argB

def dGP0(beta, epsilon, mu):
    """Return a vector of the derivatives of the 0th order
    Grand potential for a many-body system.
    """
    argA = numpy.zeros(epsilon.shape)
    for i in range(epsilon.shape[0]):
        argA[i] = dgrand_potential0(beta, epsilon[i], mu)

    return argA


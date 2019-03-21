import numpy

class Dstring(object):
    def __init__(self,n,occ):
        self.occ = numpy.asarray(occ)
        assert(n == self.occ.shape[0])
        self.n = n

    def excite(self,i,a):
        if self.occ[i] != 1:
            return (None,None)
        if self.occ[a] != 0:
            return (None,None)

        if i == a:
            return (1.0,dstring(self.n,self.occ))
        elif i < a:
            occnew = self.occ.copy()
            occnew[i] = 0
            occnew[a] = 1
            otemp = self.occ[i+1:a]
            sign = 1 if otemp.sum()%2 == 0 else -1
            return (sign,Dstring(self.n,occnew))
        elif i > a:
            occnew = self.occ.copy()
            occnew[i] = 0
            occnew[a] = 1
            otemp = self.occ[a+1:i]
            sign = 1 if otemp.sum()%2 == 0 else -1
            return (sign,dstring(self.n,occnew))

def level(ref, string):
    diff = [abs(o1 - o2) for o1,o2 in zip(ref.occ, string.occ)]
    return numpy.array(diff,dtype=int).sum()

def diff(bra, ket):
    return level(bra, ket)

def s_strings(n, nocc):
    occ = [1 if i < nocc else 0 for i in range(n)]
    ref = Dstring(n,occ)
    nvir = n - nocc
    dlist = []
    for i in range(nocc):
        for a in range(nvir):
            sign,occnew = ref.excite(i,a+nocc)
            dlist.append(occnew)
    return dlist

def d_strings(n, nocc):
    occ = [1 if i < nocc else 0 for i in range(n)]
    ref = Dstring(n,occ)
    nvir = n - nocc
    dlist = []
    for i in range(nocc):
        for a in range(nvir):
            for j in range(i+1,nocc):
                for b in range(a+1,nvir):
                    s1,d1 = ref.excite(i,a+nocc)
                    s2,d2 = d1.excite(j,b+nocc)
                    dlist.append(d2)
    return dlist

def t_strings(n, nocc):
    occ = [1 if i < nocc else 0 for i in range(n)]
    ref = Dstring(n,occ)
    nvir = n - nocc
    dlist = []
    for i in range(nocc):
        for a in range(nvir):
            for j in range(i+1,nocc):
                for b in range(a+1,nvir):
                    for k in range(j+1,nocc):
                        for c in range(b+1,nvir):
                            s1,d1 = ref.excite(i,a+nocc)
                            s2,d2 = d1.excite(j,b+nocc)
                            s3,d3 = d2.excite(k,c+nocc)
                            dlist.append(d3)
    return dlist

def q_strings(n, nocc):
    occ = [1 if i < nocc else 0 for i in range(n)]
    ref = Dstring(n,occ)
    nvir = n - nocc
    dlist = []
    for i in range(nocc):
        for a in range(nvir):
            for j in range(i+1,nocc):
                for b in range(a+1,nvir):
                    for k in range(j+1,nocc):
                        for c in range(b+1,nvir):
                            for l in range(k+1,nocc):
                                for d in range(c+1,nvir):
                                    s1,d1 = ref.excite(i,a+nocc)
                                    s2,d2 = d1.excite(j,b+nocc)
                                    s3,d3 = d2.excite(k,c+nocc)
                                    s4,d4 = d3.excite(l,d+nocc)
                                    dlist.append(d4)
    return dlist

def ci_matrixel(braa, brab, keta, ketb, ha, hb, Ia, Ib, Iabab, const):
    diffa = diff(braa, keta)//2
    diffb = diff(brab, ketb)//2
    aa = [(ob - ok) for ob,ok in zip(braa.occ,keta.occ)]
    bb = [(ob - ok) for ob,ok in zip(brab.occ,ketb.occ)]
    ao = [-1 if a < 0 else 0 for a in aa]
    av = [1 if a > 0 else 0 for a in aa]
    bo = [-1 if b < 0 else 0 for b in bb]
    bv = [1 if b > 0 else 0 for b in bb]
    if diffa == 0 and diffb == 0:
        # from Ha
        pa = braa.occ.copy()
        na = pa.sum()
        Ea = numpy.einsum('ii,i->',ha, pa)
        Ea += 0.5*numpy.einsum('ijij,i,j->',Ia, pa, pa)
        Ea -= 0.5*numpy.einsum('ijji,i,j->',Ia, pa, pa)

        # from Hb
        pb = brab.occ.copy()
        Eb = numpy.einsum('ii,i->',hb, pb)
        Eb += 0.5*numpy.einsum('ijij,i,j->',Ib, pb, pb)
        Eb -= 0.5*numpy.einsum('ijji,i,j->',Ib, pb, pb)

        # from Hab
        Eab = numpy.einsum('ijij,i,j',Iabab,pa,pb)
        Hel = Ea + Eb + Eab

        return Hel + const

    elif diffa == 1 and diffb == 0:
        o = numpy.nonzero(ao)[0][0]
        v = numpy.nonzero(av)[0][0]
        pa = keta.occ.copy()
        pb = ketb.occ.copy()
        ba = braa.occ.copy()
        i1 = min(o,v)
        i2 = max(o,v)
        sign = 1.0 if ba[i1+1:i2].sum()%2 == 0 else -1.0
        Hel = 0.0
        # from Ha
        Hel += ha[v,o]
        Hel += numpy.einsum('ii,i->',Ia[v,:,o,:],pa)
        Hel -= numpy.einsum('ii,i->',Ia[v,:,:,o],pa)
        # from Hab
        Hel += numpy.einsum('ii,i->',Iabab[v,:,o,:],pb)
        return sign*Hel

    elif diffa == 0 and diffb == 1:
        o = numpy.nonzero(bo)[0][0]
        v = numpy.nonzero(bv)[0][0]
        pa = keta.occ.copy()
        pb = ketb.occ.copy()
        bb = brab.occ.copy()
        i1 = min(o,v)
        i2 = max(o,v)
        sign = 1.0 if bb[i1+1:i2].sum()%2 == 0 else -1.0
        Hel = 0.0
        # from Hb
        Hel += hb[v,o]
        Hel += numpy.einsum('ii,i->',Ib[v,:,o,:],pb)
        Hel -= numpy.einsum('ii,i->',Ib[v,:,:,o],pb)
        # from Hab
        Hel += numpy.einsum('ii,i->',Iabab[:,v,:,o],pa)
        return sign*Hel

    elif diffa == 1 and diffb == 1:
        oa = numpy.nonzero(ao)[0][0]
        va = numpy.nonzero(av)[0][0]
        ob = numpy.nonzero(bo)[0][0]
        vb = numpy.nonzero(bv)[0][0]
        ba = braa.occ.copy()
        bb = brab.occ.copy()
        i1a = min(oa,va)
        i2a = max(oa,va)
        i1b = min(ob,vb)
        i2b = max(ob,vb)
        signa = 1.0 if ba[i1a+1:i2a].sum()%2 == 0 else -1.0
        signb = 1.0 if bb[i1b+1:i2b].sum()%2 == 0 else -1.0
        # from Hab
        Hel = Iabab[va,vb,oa,ob]
        return signa*signb*Hel

    elif diffa == 2 and diffb == 0:
        o = numpy.nonzero(ao)[0]
        v = numpy.nonzero(av)[0]
        o1,o2 = o
        v1,v2 = v
        ba = braa.occ.copy()
        i1,i2 = min(o1,v1),max(o1,v1)
        j1,j2 = min(o2,v2),max(o2,v2)
        sign1 = 1.0 if ba[i1+1:i2].sum()%2 == 0 else -1.0
        sign2 = 1.0 if ba[j1+1:j2].sum()%2 == 0 else -1.0
        # from Ha
        Hel = Ia[v1,v2,o1,o2] - Ia[v1,v2,o2,o1]
        return sign1*sign2*Hel

    elif diffa == 0 and diffb == 2:
        o = numpy.nonzero(bo)[0]
        v = numpy.nonzero(bv)[0]
        o1,o2 = o
        v1,v2 = v
        bb = brab.occ.copy()
        i1,i2 = min(o1,v1),max(o1,v1)
        j1,j2 = min(o2,v2),max(o2,v2)
        sign1 = 1.0 if bb[i1+1:i2].sum()%2 == 0 else -1.0
        sign2 = 1.0 if bb[j1+1:j2].sum()%2 == 0 else -1.0
        # from Hb
        Hel = Ib[v1,v2,o1,o2] - Ib[v1,v2,o2,o1]
        return sign1*sign2*Hel

    else:
        return 0.0

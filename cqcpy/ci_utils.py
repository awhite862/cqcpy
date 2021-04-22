import numpy


class Dstring(object):
    def __init__(self, n, occ):
        self.occ = numpy.asarray(occ)
        assert(n == self.occ.shape[0])
        self.n = n

    def excite(self, i, a):
        if self.occ[i] != 1:
            return (None,None)
        if self.occ[a] != 0:
            return (None,None)

        if i == a:
            return (1.0, Dstring(self.n, self.occ))
        elif i < a:
            occnew = self.occ.copy()
            occnew[i] = 0
            occnew[a] = 1
            otemp = self.occ[i+1:a]
            sign = 1 if otemp.sum() % 2 == 0 else -1
            return (sign, Dstring(self.n, occnew))
        elif i > a:
            occnew = self.occ.copy()
            occnew[i] = 0
            occnew[a] = 1
            otemp = self.occ[a+1:i]
            sign = 1 if otemp.sum() % 2 == 0 else -1
            return (sign, Dstring(self.n, occnew))

    def __eq__(self, other):
        return numpy.array_equal(self.occ, other.occ) and self.n == other.n


class Pstring(object):
    def __init__(self, n, occ):
        self.occ = numpy.asarray(occ)
        assert(n == self.occ.shape[0])
        self.n = n

    def raize(self, p):
        occnew = self.occ.copy()
        occnew[p] = occnew[p] + 1
        return Pstring(self.n,occnew)

    def lower(self, p):
        occnew = self.occ.copy()
        if occnew[p] == 0:
            return None
        else:
            occnew[p] = occnew[p] - 1
            return Pstring(self.n, occnew)

    def __eq__(self, other):
        return numpy.array_equal(self.occ, other.occ) and self.n == other.n


def level(ref, string):
    diff = [abs(o1 - o2) for o1,o2 in zip(ref.occ, string.occ)]
    return numpy.array(diff, dtype=int).sum()


def diff(bra, ket):
    return level(bra, ket)


def s_strings(n, nocc, occ=None):
    if occ is None: occ = [1 if i < nocc else 0 for i in range(n)]
    vir = [1 if x == 0 else 0 for x in occ]
    iocc = [i for i,x in enumerate(occ) if x > 0]
    ivir = [i for i,x in enumerate(vir) if x > 0]
    ref = Dstring(n, occ)
    dlist = []
    for i in iocc:
        for a in ivir:
            sign, occnew = ref.excite(i, a)
            dlist.append(occnew)
    return dlist


def d_strings(n, nocc, occ=None):
    if occ is None: occ = [1 if i < nocc else 0 for i in range(n)]
    vir = [1 if x == 0 else 0 for x in occ]
    iocc = [i for i,x in enumerate(occ) if x > 0]
    ivir = [i for i,x in enumerate(vir) if x > 0]
    ref = Dstring(n, occ)
    dlist = []
    for i in iocc:
        for a in ivir:
            for j in iocc:
                if j <= i: continue
                for b in ivir:
                    if b <= a: continue
                    s1, d1 = ref.excite(i, a)
                    s2, d2 = d1.excite(j, b)
                    dlist.append(d2)
    return dlist


def t_strings(n, nocc, occ=None):
    if occ is None: occ = [1 if i < nocc else 0 for i in range(n)]
    vir = [1 if x == 0 else 0 for x in occ]
    iocc = [i for i,x in enumerate(occ) if x > 0]
    ivir = [i for i,x in enumerate(vir) if x > 0]
    ref = Dstring(n, occ)
    dlist = []
    for i in iocc:
        for a in ivir:
            for j in iocc:
                if j <= i: continue
                for b in ivir:
                    if b <= a: continue
                    for k in iocc:
                        if k <= j: continue
                        for c in ivir:
                            if c <= b: continue
                            s1, d1 = ref.excite(i, a)
                            s2, d2 = d1.excite(j, b)
                            s3, d3 = d2.excite(k, c)
                            dlist.append(d3)
    return dlist


def q_strings(n, nocc, occ=None):
    if occ is None: occ = [1 if i < nocc else 0 for i in range(n)]
    vir = [1 if x == 0 else 0 for x in occ]
    iocc = [i for i,x in enumerate(occ) if x > 0]
    ivir = [i for i,x in enumerate(vir) if x > 0]
    ref = Dstring(n, occ)
    dlist = []
    for i in occ:
        for a in ivir:
            for j in occ:
                if j <= i: continue
                for b in ivir:
                    if b <= a: continue
                    for k in iocc:
                        if k <= j: continue
                        for c in ivir:
                            if c <= b: continue
                            for l in iocc:
                                if l <= k: continue
                                for d in ivir:
                                    if d <= c: continue
                                    s1, d1 = ref.excite(i, a)
                                    s2, d2 = d1.excite(j, b)
                                    s3, d3 = d2.excite(k, c)
                                    s4, d4 = d3.excite(l, d)
                                    dlist.append(d4)
    return dlist


def s_pstrings(nmode):
    occ = [0 for i in range(nmode)]
    ref = Pstring(nmode, occ)
    blist = []
    for i in range(nmode):
        temp = ref.raize(i)
        blist.append(temp)
    return blist


def d_pstrings(nmode):
    occ = [0 for i in range(nmode)]
    ref = Pstring(nmode, occ)
    blist = []
    for i in range(nmode):
        for j in range(i, nmode):
            temp = ref.raize(i).raize(j)
            blist.append(temp)
    return blist


def ucis_basis(n, na, nb, gs=True):
    sa = s_strings(n, na)
    sb = s_strings(n, nb)
    occa = [1 if i < na else 0 for i in range(n)]
    occb = [1 if i < nb else 0 for i in range(n)]
    refa = Dstring(n, occa)
    refb = Dstring(n, occb)
    if gs:
        basis = [(refa,refb)]
    else:
        basis = []
    for a in sa:
        basis.append((a,refb))
    for b in sb:
        basis.append((refa,b))
    return basis


def ucisd_basis(n, na, nb):
    sa = s_strings(n, na)
    sb = s_strings(n, nb)
    da = d_strings(n, na)
    db = d_strings(n, nb)
    occa = [1 if i < na else 0 for i in range(n)]
    occb = [1 if i < nb else 0 for i in range(n)]
    refa = Dstring(n, occa)
    refb = Dstring(n, occb)
    basis = [(refa,refb)]
    for a in sa:
        basis.append((a,refb))
    for b in sb:
        basis.append((refa,b))
    for a in da:
        basis.append((a,refb))
    for b in db:
        basis.append((refa,b))
    for a in sa:
        for b in sb:
            basis.append((a,b))
    return basis


def ucisdt_basis(n, na, nb):
    sa = s_strings(n, na)
    sb = s_strings(n, nb)
    da = d_strings(n, na)
    db = d_strings(n, nb)
    ta = t_strings(n, na)
    tb = t_strings(n, nb)
    occa = [1 if i < na else 0 for i in range(n)]
    occb = [1 if i < nb else 0 for i in range(n)]
    refa = Dstring(n, occa)
    refb = Dstring(n, occb)
    basis = [(refa,refb)]
    for a in sa:
        basis.append((a,refb))
    for b in sb:
        basis.append((refa,b))
    for a in da:
        basis.append((a,refb))
    for b in db:
        basis.append((refa,b))
    for a in sa:
        for b in sb:
            basis.append((a,b))
    for a in ta:
        basis.append((a,refb))
    for b in tb:
        basis.append((refa,b))
    for a in da:
        for b in sb:
            basis.append((a,b))
    for a in sa:
        for b in db:
            basis.append((a,b))
    return basis


def ucisdtq_basis(n, na, nb):
    sa = s_strings(n, na)
    sb = s_strings(n, nb)
    da = d_strings(n, na)
    db = d_strings(n, nb)
    ta = t_strings(n, na)
    tb = t_strings(n, nb)
    qa = q_strings(n, na)
    qb = q_strings(n, nb)
    occa = [1 if i < na else 0 for i in range(n)]
    occb = [1 if i < nb else 0 for i in range(n)]
    refa = Dstring(n, occa)
    refb = Dstring(n, occb)
    basis = [(refa,refb)]
    for a in sa:
        basis.append((a,refb))
    for b in sb:
        basis.append((refa,b))
    for a in da:
        basis.append((a,refb))
    for b in db:
        basis.append((refa,b))
    for a in sa:
        for b in sb:
            basis.append((a,b))
    for a in ta:
        basis.append((a,refb))
    for b in tb:
        basis.append((refa,b))
    for a in da:
        for b in sb:
            basis.append((a,b))
    for a in sa:
        for b in db:
            basis.append((a,b))
    for a in qa:
        basis.append((a,refb))
    for b in qb:
        basis.append((refa,b))
    for a in ta:
        for b in sb:
            basis.append((a,b))
    for a in da:
        for b in db:
            basis.append((a,b))
    for a in sa:
        for b in tb:
            basis.append((a,b))
    return basis


def gcis_basis(nmo, n, gs=True):
    nb = n//2
    na = n - nb
    occa = [1 if i < na else 0 for i in range(nmo//2)]
    occb = [1 if i < nb else 0 for i in range(nmo//2)]
    occ = occa + occb
    ref = Dstring(nmo, occ)
    singles = s_strings(nmo, n, occ=occ)
    basis = [ref] if gs else []
    for s in singles:
        basis.append(s)
    return basis


def gcisd_basis(nmo, n):
    nb = n//2
    na = n - n//2
    occa = [1 if i < na else 0 for i in range(nmo//2)]
    occb = [1 if i < nb else 0 for i in range(nmo//2)]
    occ = occa + occb
    ref = Dstring(nmo, occ)
    singles = s_strings(nmo, n, occ=occ)
    doubles = d_strings(nmo, n, occ=occ)
    basis = [ref]
    for s in singles:
        basis.append(s)
    for d in doubles:
        basis.append(d)
    return basis


def gcisdt_basis(nmo, n):
    nb = n//2
    na = n - n//2
    occa = [1 if i < na else 0 for i in range(nmo//2)]
    occb = [1 if i < nb else 0 for i in range(nmo//2)]
    occ = occa + occb
    ref = Dstring(nmo, occ)
    singles = s_strings(nmo, n, occ=occ)
    doubles = d_strings(nmo, n, occ=occ)
    triples = t_strings(nmo, n, occ=occ)
    basis = [ref]
    for s in singles:
        basis.append(s)
    for d in doubles:
        basis.append(d)
    for t in triples:
        basis.append(t)
    return basis


def gcisdtq_basis(nmo, n):
    nb = n//2
    na = n - n//2
    occa = [1 if i < na else 0 for i in range(nmo//2)]
    occb = [1 if i < nb else 0 for i in range(nmo//2)]
    occ = occa + occb
    ref = Dstring(nmo, occ)
    singles = s_strings(nmo, n, occ=occ)
    doubles = d_strings(nmo, n, occ=occ)
    triples = t_strings(nmo, n, occ=occ)
    quadles = q_strings(nmo, n, occ=occ)
    basis = [ref]
    for s in singles:
        basis.append(s)
    for d in doubles:
        basis.append(d)
    for t in triples:
        basis.append(t)
    for q in quadles:
        basis.append(q)
    return basis


def vcis_basis(nmode):
    occ = [0 for i in range(nmode)]
    ref = Pstring(nmode, occ)
    basis = [ref]
    singles = s_pstrings(nmode)
    for s in singles:
        basis.append(s)
    return basis


def vcisd_basis(nmode):
    occ = [0 for i in range(nmode)]
    ref = Pstring(nmode, occ)
    basis = [ref]
    singles = s_pstrings(nmode)
    doubles = d_pstrings(nmode)
    for s in singles:
        basis.append(s)
    for d in doubles:
        basis.append(d)
    return basis


def gmakeCfromT(no, nv, T1, T2, order=2, occ=None):
    nmo = no + nv
    if order < 1:
        raise Exception("Unrecognized CI expansion order: {}".format(order))
    if order == 1:
        basis = gcis_basis(nmo, no)
    elif order == 2:
        basis = gcisd_basis(nmo, no)
    elif order == 3:
        basis = gcisdt_basis(nmo, no)
    elif order == 4:
        basis = gcisdtq_basis(nmo, no)
    else:
        raise Exception("Higher than 4th order is not supported")

    if occ is None: occ = [1 if i < no else 0 for i in range(nmo//2)]
    vir = [1 if x == 0 else 0 for x in occ]
    iocc = [i for i,x in enumerate(occ) if x > 0]
    ivir = [i for i,x in enumerate(vir) if x > 0]
    ref = Dstring(nmo, occ)

    nd = len(basis)
    C = numpy.zeros(nd)
    C[0] = 1.0
    if order >= 1:
        # loop over T1
        for ii,i in enumerate(iocc):
            for ia,a in enumerate(ivir):
                s, dstr = ref.excite(i, a)
                if dstr is not None:
                    idx = basis.index(dstr)
                    C[idx] += s*T1[ia,ii]
    if order >= 2:
        # loop over T2
        for ii,i in enumerate(iocc):
            for ij,j in enumerate(iocc):
                if j <= i: continue
                for ia,a in enumerate(ivir):
                    for ib,b in enumerate(ivir):
                        if b <= a: continue
                        s1, dstr = ref.excite(i, a)
                        if dstr is None:
                            continue
                        s2, dstr = dstr.excite(j, b)
                        if dstr is not None:
                            idx = basis.index(dstr)
                            C[idx] += s1*s2*T2[ia,ib,ii,ij]
        # loop over T1^2
        for ii,i in enumerate(iocc):
            for ij,j in enumerate(iocc):
                for ia,a in enumerate(ivir):
                    for ib,b in enumerate(ivir):
                        s1, dstr = ref.excite(i, a)
                        if dstr is None:
                            continue
                        s2, dstr = dstr.excite(j, b)
                        if dstr is not None:
                            idx = basis.index(dstr)
                            C[idx] += 0.5*s1*s2*T1[ia,ii]*T1[ib,ij]
    if order >= 3:
        # loop over T1T2
        for ii,i in enumerate(iocc):
            for ij,j in enumerate(iocc):
                for ik,k in enumerate(iocc):
                    if k <= j: continue
                    for ia,a in enumerate(ivir):
                        for ib,b in enumerate(ivir):
                            for ic,c in enumerate(ivir):
                                if c <= b: continue
                                s1, dstr = ref.excite(i, a)
                                if dstr is None:
                                    continue
                                s2, dstr = dstr.excite(j, b)
                                if dstr is None:
                                    continue
                                s3, dstr = dstr.excite(k, c)
                                if dstr is not None:
                                    idx = basis.index(dstr)
                                    C[idx] += s1*s2*s3*T1[ia,ii]*T2[ib,ic,ij,ik]
        # loop over T1^3
        for ii,i in enumerate(iocc):
            for ij,j in enumerate(iocc):
                for ik,k in enumerate(iocc):
                    for ia,a in enumerate(ivir):
                        for ib,b in enumerate(ivir):
                            for ic,c in enumerate(ivir):
                                s1, dstr = ref.excite(i, a)
                                if dstr is None:
                                    continue
                                s2, dstr = dstr.excite(j, b)
                                if dstr is None:
                                    continue
                                s3, dstr = dstr.excite(k, c)
                                if dstr is not None:
                                    idx = basis.index(dstr)
                                    C[idx] += s1*s2*s3*T1[ia,ii]*T1[ib,ij]*T1[ic,ik]/6.0
    if order >= 4:
        # T2^2
        for ii,i in enumerate(iocc):
            for ij,j in enumerate(iocc):
                if j <= i: continue
                for ik,k in enumerate(iocc):
                    for il,l in enumerate(iocc):
                        if l <= k: continue
                        for ia,a in enumerate(ivir):
                            for ib,b in enumerate(ivir):
                                if b <= a: continue
                                for ic,c in enumerate(ivir):
                                    for idd,d in enumerate(ivir):
                                        if d <= c: continue
                                        s1, dstr = ref.excite(i, a)
                                        if dstr is None:
                                            continue
                                        s2, dstr = dstr.excite(j, b)
                                        if dstr is None:
                                            continue
                                        s3, dstr = dstr.excite(k, c)
                                        if dstr is None:
                                            continue
                                        s4, dstr = dstr.excite(l, d)
                                        if dstr is not None:
                                            idx = basis.index(dstr)
                                            C[idx] += s1*s2*s3*s4*T2[ia,ib,ii,ij]*T2[ic,idd,ik,il]/2.0
        # T1^2T2
        for ii,i in enumerate(iocc):
            for ij,j in enumerate(iocc):
                for ik,k in enumerate(iocc):
                    for il,l in enumerate(iocc):
                        if l <= k: continue
                        for ia,a in enumerate(ivir):
                            for ib,b in enumerate(ivir):
                                for ic,c in enumerate(ivir):
                                    for idd,d in enumerate(ivir):
                                        if d <= c: continue
                                        s1, dstr = ref.excite(i, a)
                                        if dstr is None:
                                            continue
                                        s2, dstr = dstr.excite(j, b)
                                        if dstr is None:
                                            continue
                                        s3, dstr = dstr.excite(k, c)
                                        if dstr is None:
                                            continue
                                        s4, dstr = dstr.excite(l, d)
                                        if dstr is not None:
                                            idx = basis.index(dstr)
                                            C[idx] += s1*s2*s3*s4*T1[ia,ii]*T1[ib,ij]*T2[ic,idd,ik,il]/2.0
        for ii,i in enumerate(iocc):
            for ij,j in enumerate(iocc):
                for ik,k in enumerate(iocc):
                    for il,l in enumerate(iocc):
                        for ia,a in enumerate(ivir):
                            for ib,b in enumerate(ivir):
                                for ic,c in enumerate(ivir):
                                    for idd,d in enumerate(ivir):
                                        s1, dstr = ref.excite(i, a)
                                        if dstr is None:
                                            continue
                                        s2, dstr = dstr.excite(j, b)
                                        if dstr is None:
                                            continue
                                        s3, dstr = dstr.excite(k, c)
                                        if dstr is None:
                                            continue
                                        s4, dstr = dstr.excite(l, d)
                                        if dstr is not None:
                                            idx = basis.index(dstr)
                                            # T1^4
                                            C[idx] += s1*s2*s3*s4*T1[ia,ii]*T1[ib,ij]*T1[ic,ik]*T1[idd,il]/24.0
    return C


def makeCfromT(noa, nva, nob, nvb, T1a, T1b, T2aa, T2ab, T2bb, order=2):
    na = noa + nva
    nb = nob + nvb
    assert(na == nb)
    if order < 1:
        raise Exception("Unrecognized CI expansion order: {}".format(order))
    if order == 1:
        basis = ucis_basis(na, noa, nob)
    elif order == 2:
        basis = ucisd_basis(na, noa, nob)
    elif order == 3:
        basis = ucisdt_basis(na, noa, nob)
    elif order == 4:
        basis = ucisdtq_basis(na, noa, nob)
    else:
        raise Exception("Higher than 4th order is not supported")

    occa = [1 if i < noa else 0 for i in range(na)]
    occb = [1 if i < nob else 0 for i in range(nb)]
    refa = Dstring(na, occa)
    refb = Dstring(nb, occb)
    nd = len(basis)
    C = numpy.zeros(nd)
    C[0] = 1.0

    if order >= 1:
        # loop over T1a
        for i in range(noa):
            for a in range(nva):
                s, astr = refa.excite(i, a + noa)
                if astr is not None:
                    idx = basis.index((astr,refb))
                    C[idx] += s*T1a[a,i]
        # loop over T1b
        for i in range(nob):
            for a in range(nvb):
                s, bstr = refb.excite(i, a + nob)
                if bstr is not None:
                    idx = basis.index((refa,bstr))
                    C[idx] += s*T1b[a,i]

    if order >= 2:
        # loop over T2aa
        for i in range(noa):
            for j in range(i+1, noa):
                for a in range(nva):
                    for b in range(a+1, nva):
                        s1, astr = refa.excite(i, a + noa)
                        if astr is None:
                            continue
                        s2, astr = astr.excite(j, b + noa)
                        if astr is not None:
                            idx = basis.index((astr,refb))
                            C[idx] += s1*s2*T2aa[a,b,i,j]
        # loop over T2bb
        for i in range(nob):
            for j in range(i+1, nob):
                for a in range(nvb):
                    for b in range(a+1, nvb):
                        s1, bstr = refb.excite(i, a + nob)
                        if bstr is None:
                            continue
                        s2, bstr = bstr.excite(j, b + nob)
                        if bstr is not None:
                            idx = basis.index((refa,bstr))
                            C[idx] += s1*s2*T2bb[a,b,i,j]
        # loop over T2ab
        for i in range(noa):
            for j in range(nob):
                for a in range(nva):
                    for b in range(nvb):
                        sa, astr = refa.excite(i, a + noa)
                        sb, bstr = refb.excite(j, b + nob)
                        if astr is not None and bstr is not None:
                            idx = basis.index((astr,bstr))
                            C[idx] += sa*sb*T2ab[a,b,i,j]

        # loop over T1a^2
        for i in range(noa):
            for j in range(noa):
                for a in range(nva):
                    for b in range(nva):
                        s1, astr = refa.excite(i, a + noa)
                        if astr is None:
                            continue
                        s2, astr = astr.excite(j, b + noa)
                        if astr is not None:
                            idx = basis.index((astr,refb))
                            C[idx] += 0.5*s1*s2*T1a[a,i]*T1a[b,j]
        # loop over T1b^2
        for i in range(nob):
            for j in range(nob):
                for a in range(nvb):
                    for b in range(nvb):
                        s1, bstr = refb.excite(i, a + nob)
                        if bstr is None:
                            continue
                        s2, bstr = bstr.excite(j, b + nob)
                        if bstr is not None:
                            idx = basis.index((refa,bstr))
                            C[idx] += 0.5*s1*s2*T1b[a,i]*T1b[b,j]
        # loop over T1a*T1b
        for i in range(noa):
            for j in range(nob):
                for a in range(nva):
                    for b in range(nvb):
                        sa, astr = refa.excite(i, a + noa)
                        sb, bstr = refb.excite(j, b + nob)
                        if astr is not None and bstr is not None:
                            idx = basis.index((astr,bstr))
                            C[idx] += sa*sb*T1a[a,i]*T1b[b,j]
    if order >= 3:
        # loop over T1a*T2aa
        for i in range(noa):
            for j in range(noa):
                for k in range(j+1, noa):
                    for a in range(nva):
                        for b in range(nva):
                            for c in range(b+1, nva):
                                s1, astr = refa.excite(i, a + noa)
                                if astr is None:
                                    continue
                                s2, astr = astr.excite(j, b + noa)
                                if astr is None:
                                    continue
                                s3, astr = astr.excite(k, c + noa)
                                if astr is not None:
                                    idx = basis.index((astr,refb))
                                    C[idx] += s1*s2*s3*T1a[a,i]*T2aa[b,c,j,k]
        # loop over T1a*T2ab
        for i in range(noa):
            for j in range(noa):
                for k in range(nob):
                    for a in range(nva):
                        for b in range(nva):
                            for c in range(nvb):
                                s1a, astr = refa.excite(i, a + noa)
                                if astr is None:
                                    continue
                                s2a, astr = astr.excite(j, b + noa)
                                sb, bstr = refb.excite(k, c + nob)
                                if astr is not None and bstr is not None:
                                    idx = basis.index((astr,bstr))
                                    C[idx] += s1a*s2a*sb*T1a[a,i]*T2ab[b,c,j,k]
        # loop over T1b*T2aa
        for i in range(nob):
            for j in range(noa):
                for k in range(j+1, noa):
                    for a in range(nvb):
                        for b in range(nva):
                            for c in range(b+1, nva):
                                sb, bstr = refb.excite(i, a + nob)
                                s1a, astr = refa.excite(j, b + noa)
                                if astr is None:
                                    continue
                                s2a, astr = astr.excite(k, c + noa)
                                if astr is not None and bstr is not None:
                                    idx = basis.index((astr,bstr))
                                    C[idx] += s1a*s2a*sb*T1b[a,i]*T2aa[b,c,j,k]
        # loop over T1a*T2bb
        for i in range(noa):
            for j in range(nob):
                for k in range(j+1, nob):
                    for a in range(nva):
                        for b in range(nvb):
                            for c in range(b+1, nvb):
                                sa, astr = refa.excite(i, a + noa)
                                s1b, bstr = refb.excite(j, b + nob)
                                if bstr is None:
                                    continue
                                s2b, bstr = bstr.excite(k, c + nob)
                                if astr is not None and bstr is not None:
                                    idx = basis.index((astr,bstr))
                                    C[idx] += s1b*s2b*sa*T1a[a,i]*T2bb[b,c,j,k]
        # loop over T1b*T2ab
        for i in range(nob):
            for j in range(noa):
                for k in range(nob):
                    for a in range(nvb):
                        for b in range(nva):
                            for c in range(nvb):
                                s1b, bstr = refb.excite(i, a + nob)
                                if bstr is None:
                                    continue
                                sa, astr = refa.excite(j, b + noa)
                                s2b, bstr = bstr.excite(k, c + nob)
                                if astr is not None and bstr is not None:
                                    idx = basis.index((astr,bstr))
                                    C[idx] += s1b*s2b*sa*T1b[a,i]*T2ab[b,c,j,k]
        # loop over T1b*T2bb
        for i in range(nob):
            for j in range(nob):
                for k in range(j+1, nob):
                    for a in range(nvb):
                        for b in range(nvb):
                            for c in range(b+1, nvb):
                                s1, bstr = refb.excite(i, a + nob)
                                if bstr is None:
                                    continue
                                s2, bstr = bstr.excite(j, b + nob)
                                if bstr is None:
                                    continue
                                s3, bstr = bstr.excite(k, c + nob)
                                if bstr is not None:
                                    idx = basis.index((refa,bstr))
                                    C[idx] += s1*s2*s3*T1b[a,i]*T2bb[b,c,j,k]

        # loop over T1a^3
        for i in range(noa):
            for j in range(noa):
                for k in range(noa):
                    for a in range(nva):
                        for b in range(nva):
                            for c in range(nva):
                                s1, astr = refa.excite(i, a + noa)
                                if astr is None:
                                    continue
                                s2, astr = astr.excite(j, b + noa)
                                if astr is None:
                                    continue
                                s3, astr = astr.excite(k, c + noa)
                                if astr is not None:
                                    idx = basis.index((astr,refb))
                                    C[idx] += s1*s2*s3*T1a[a,i]*T1a[b,j]*T1a[c,k]/6.0
        # loop over T1a^2T1b
        for i in range(noa):
            for j in range(noa):
                for k in range(nob):
                    for a in range(nva):
                        for b in range(nva):
                            for c in range(nvb):
                                s1a,astr = refa.excite(i, a + noa)
                                if astr is None:
                                    continue
                                s2a, astr = astr.excite(j, b + noa)
                                sb, bstr = refb.excite(k, c + nob)
                                if astr is not None and bstr is not None:
                                    idx = basis.index((astr,bstr))
                                    C[idx] += s1a*s2a*sb*T1a[a,i]*T1a[b,j]*T1b[c,k]/2.0
        # loop over T1aT1b^2
        for i in range(noa):
            for j in range(nob):
                for k in range(nob):
                    for a in range(nva):
                        for b in range(nvb):
                            for c in range(nvb):
                                sa, astr = refa.excite(i, a + noa)
                                s1b, bstr = refb.excite(j, b + nob)
                                if bstr is None:
                                    continue
                                s2b, bstr = bstr.excite(k, c + nob)
                                if astr is not None and bstr is not None:
                                    idx = basis.index((astr,bstr))
                                    C[idx] += s1b*s2b*sa*T1a[a,i]*T1b[b,j]*T1b[c,k]/2.0
        # loop over T1b^3
        for i in range(nob):
            for j in range(nob):
                for k in range(nob):
                    for a in range(nvb):
                        for b in range(nvb):
                            for c in range(nvb):
                                s1, bstr = refb.excite(i, a + nob)
                                if bstr is None:
                                    continue
                                s2, bstr = bstr.excite(j, b + nob)
                                if bstr is None:
                                    continue
                                s3, bstr = bstr.excite(k, c + nob)
                                if bstr is not None:
                                    idx = basis.index((refa,bstr))
                                    C[idx] += s1*s2*s3*T1b[a,i]*T1b[b,j]*T1b[c,k]/6.0
    if order >= 4:
        # loop over T2aa^2
        for i in range(noa):
            for j in range(i+1, noa):
                for k in range(noa):
                    for l in range(k+1, noa):
                        for a in range(nva):
                            for b in range(a+1, nva):
                                for c in range(nva):
                                    for d in range(c+1, nva):
                                        sa1, astr = refa.excite(i, a + noa)
                                        if astr is None:
                                            continue
                                        sa2, astr = astr.excite(j, b + noa)
                                        if astr is None:
                                            continue
                                        sa3, astr = astr.excite(k, c + noa)
                                        if astr is None:
                                            continue
                                        sa4, astr = astr.excite(l, d + noa)
                                        if astr is not None:
                                            idx = basis.index((astr,refb))
                                            C[idx] += sa1*sa2*sa3*sa4*T2aa[a,b,i,j]*T2aa[c,d,k,l]/2.0
        # loop over T2aa*T2ab
        for i in range(noa):
            for j in range(i+1, noa):
                for k in range(noa):
                    for l in range(nob):
                        for a in range(nva):
                            for b in range(a+1, nva):
                                for c in range(nva):
                                    for d in range(nvb):
                                        sa1, astr = refa.excite(i, a + noa)
                                        if astr is None:
                                            continue
                                        sa2, astr = astr.excite(j, b + noa)
                                        if astr is None:
                                            continue
                                        sa3, astr = astr.excite(k, c + noa)
                                        sb1, bstr = refb.excite(l, d + nob)
                                        if bstr is not None and astr is not None:
                                            idx = basis.index((astr,bstr))
                                            C[idx] += sa1*sa2*sa3*sb1*T2aa[a,b,i,j]*T2ab[c,d,k,l]
        # loop over T2aa*T2bb
        for i in range(noa):
            for j in range(i+1, noa):
                for k in range(nob):
                    for l in range(k+1, nob):
                        for a in range(nva):
                            for b in range(a+1, nva):
                                for c in range(nvb):
                                    for d in range(c+1, nvb):
                                        sa1, astr = refa.excite(i, a + noa)
                                        if astr is None:
                                            continue
                                        sa2, astr = astr.excite(j, b + noa)
                                        sb1, bstr = refb.excite(k, c + nob)
                                        if bstr is None:
                                            continue
                                        sb2, bstr = bstr.excite(l, d + nob)
                                        if bstr is not None and astr is not None:
                                            idx = basis.index((astr,bstr))
                                            C[idx] += sa1*sa2*sb1*sb2*T2aa[a,b,i,j]*T2bb[c,d,k,l]
        # loop over T2ab^2
        for i in range(noa):
            for j in range(nob):
                for k in range(noa):
                    for l in range(nob):
                        for a in range(nva):
                            for b in range(nvb):
                                for c in range(nva):
                                    for d in range(nvb):
                                        sa1, astr = refa.excite(i, a + noa)
                                        if astr is None:
                                            continue
                                        sa2, astr = astr.excite(k, c + noa)
                                        sb1, bstr = refb.excite(j, b + nob)
                                        if bstr is None:
                                            continue
                                        sb2, bstr = bstr.excite(l, d + nob)
                                        if bstr is not None and astr is not None:
                                            idx = basis.index((astr,bstr))
                                            C[idx] += sa1*sa2*sb1*sb2*T2ab[a,b,i,j]*T2ab[c,d,k,l]/2.0
        # loop over T2ab*T2bb
        for i in range(noa):
            for j in range(nob):
                for k in range(nob):
                    for l in range(k+1, nob):
                        for a in range(nva):
                            for b in range(nvb):
                                for c in range(nvb):
                                    for d in range(c+1, nvb):
                                        sa, astr = refa.excite(i, a + noa)
                                        s1, bstr = refb.excite(j, b + nob)
                                        if bstr is None:
                                            continue
                                        s2, bstr = bstr.excite(k, c + nob)
                                        if bstr is None:
                                            continue
                                        s3, bstr = bstr.excite(l, d + nob)
                                        if bstr is not None and astr is not None:
                                            idx = basis.index((astr,bstr))
                                            C[idx] += sa*s1*s2*s3*T2ab[a,b,i,j]*T2bb[c,d,k,l]
        # loop over T2bb^2
        for i in range(nob):
            for j in range(i+1, nob):
                for k in range(nob):
                    for l in range(k+1, nob):
                        for a in range(nvb):
                            for b in range(a+1, nvb):
                                for c in range(nvb):
                                    for d in range(c+1, nvb):
                                        s1, bstr = refb.excite(i, a + nob)
                                        if bstr is None:
                                            continue
                                        s2, bstr = bstr.excite(j, b + nob)
                                        if bstr is None:
                                            continue
                                        s3, bstr = bstr.excite(k, c + nob)
                                        if bstr is None:
                                            continue
                                        s4, bstr = bstr.excite(l, d + nob)
                                        if bstr is not None:
                                            idx = basis.index((refa,bstr))
                                            C[idx] += s1*s2*s3*s4*T2bb[a,b,i,j]*T2bb[c,d,k,l]/2.0

        # loop over T1a^2*T2aa
        for i in range(noa):
            for j in range(noa):
                for k in range(noa):
                    for l in range(k+1, noa):
                        for a in range(nva):
                            for b in range(nva):
                                for c in range(nva):
                                    for d in range(c+1, nva):
                                        sa1, astr = refa.excite(i, a + noa)
                                        if astr is None:
                                            continue
                                        sa2, astr = astr.excite(j, b + noa)
                                        if astr is None:
                                            continue
                                        sa3, astr = astr.excite(k, c + noa)
                                        if astr is None:
                                            continue
                                        sa4, astr = astr.excite(l, d + noa)
                                        if astr is not None:
                                            idx = basis.index((astr,refb))
                                            C[idx] += sa1*sa2*sa3*sa4*T1a[a,i]*T1a[b,j]*T2aa[c,d,k,l]/2.0
        # loop over T1a^2*T2ab
        for i in range(noa):
            for j in range(noa):
                for k in range(noa):
                    for l in range(nob):
                        for a in range(nva):
                            for b in range(nva):
                                for c in range(nva):
                                    for d in range(nvb):
                                        sa1, astr = refa.excite(i, a + noa)
                                        if astr is None:
                                            continue
                                        sa2, astr = astr.excite(j, b + noa)
                                        if astr is None:
                                            continue
                                        sa3, astr = astr.excite(k, c + noa)
                                        sb1, bstr = refb.excite(l, d + nob)
                                        if bstr is not None and astr is not None:
                                            idx = basis.index((astr,bstr))
                                            C[idx] += sa1*sa2*sa3*sb1*T1a[a,i]*T1a[b,j]*T2ab[c,d,k,l]/2.0
        # loop over T1a*T1b*T2aa
        for i in range(noa):
            for j in range(nob):
                for k in range(noa):
                    for l in range(k+1, noa):
                        for a in range(nva):
                            for b in range(nvb):
                                for c in range(nva):
                                    for d in range(c+1, nva):
                                        sa1, astr = refa.excite(i, a + noa)
                                        if astr is None:
                                            continue
                                        sa2, astr = astr.excite(k, c + noa)
                                        if astr is None:
                                            continue
                                        sa3, astr = astr.excite(l, d + noa)
                                        sb1, bstr = refb.excite(j, b + nob)
                                        if astr is not None and bstr is not None:
                                            idx = basis.index((astr,bstr))
                                            C[idx] += sa1*sa2*sa3*sb1*T1a[a,i]*T1b[b,j]*T2aa[c,d,k,l]
        # loop over T1a^2*T2bb
        for i in range(noa):
            for j in range(noa):
                for k in range(nob):
                    for l in range(k+1, nob):
                        for a in range(nva):
                            for b in range(nva):
                                for c in range(nvb):
                                    for d in range(c+1, nvb):
                                        sa1, astr = refa.excite(i, a + noa)
                                        if astr is None:
                                            continue
                                        sa2, astr = astr.excite(j, b + noa)
                                        if astr is None:
                                            continue
                                        sb1, bstr = refb.excite(k, c + nob)
                                        if bstr is None:
                                            continue
                                        sb2, bstr = bstr.excite(l, d + nob)
                                        if bstr is not None and astr is not None:
                                            idx = basis.index((astr,bstr))
                                            C[idx] += sa1*sa2*sb1*sb2*T1a[a,i]*T1a[b,j]*T2bb[c,d,k,l]/2.0
        # loop over T1a*T1b*T2ab
        for i in range(noa):
            for j in range(nob):
                for k in range(noa):
                    for l in range(nob):
                        for a in range(nva):
                            for b in range(nvb):
                                for c in range(nva):
                                    for d in range(nvb):
                                        sa1, astr = refa.excite(i, a + noa)
                                        if astr is None:
                                            continue
                                        sa2, astr = astr.excite(k, c + noa)
                                        sb1, bstr = refb.excite(j, b + nob)
                                        if bstr is None:
                                            continue
                                        sb2, bstr = bstr.excite(l, d + nob)
                                        if astr is not None and bstr is not None:
                                            idx = basis.index((astr,bstr))
                                            C[idx] += sa1*sa2*sb1*sb2*T1a[a,i]*T1b[b,j]*T2ab[c,d,k,l]
        # loop over T1a*T1b*T2bb
        for i in range(noa):
            for j in range(nob):
                for k in range(nob):
                    for l in range(k+1, nob):
                        for a in range(nva):
                            for b in range(nvb):
                                for c in range(nvb):
                                    for d in range(c+1, nvb):
                                        sa, astr = refa.excite(i, a + noa)
                                        s1, bstr = refb.excite(j, b + nob)
                                        if bstr is None:
                                            continue
                                        s2, bstr = bstr.excite(k, c + nob)
                                        if bstr is None:
                                            continue
                                        s3, bstr = bstr.excite(l, d + nob)
                                        if bstr is not None and astr is not None:
                                            idx = basis.index((astr,bstr))
                                            C[idx] += sa*s1*s2*s3*T1a[a,i]*T1b[b,j]*T2bb[c,d,k,l]
        # loop over T1b^2*T2ab
        for i in range(nob):
            for j in range(nob):
                for k in range(noa):
                    for l in range(nob):
                        for a in range(nvb):
                            for b in range(nvb):
                                for c in range(nva):
                                    for d in range(nvb):
                                        sa, astr = refa.excite(k, c + noa)
                                        s1, bstr = refb.excite(i, a + nob)
                                        if bstr is None:
                                            continue
                                        s2, bstr = bstr.excite(j, b + nob)
                                        if bstr is None:
                                            continue
                                        s3, bstr = bstr.excite(l, d + nob)
                                        if bstr is not None and astr is not None:
                                            idx = basis.index((astr,bstr))
                                            C[idx] += s1*s2*s3*sa*T1b[a,i]*T1b[b,j]*T2ab[c,d,k,l]/2.0
        # loop over T1b^2*T2bb
        for i in range(nob):
            for j in range(nob):
                for k in range(nob):
                    for l in range(k+1, nob):
                        for a in range(nvb):
                            for b in range(nvb):
                                for c in range(nvb):
                                    for d in range(c+1, nvb):
                                        s1, bstr = refb.excite(i, a + nob)
                                        if bstr is None:
                                            continue
                                        s2, bstr = bstr.excite(j, b + nob)
                                        if bstr is None:
                                            continue
                                        s3, bstr = bstr.excite(k, c + nob)
                                        if bstr is None:
                                            continue
                                        s4, bstr = bstr.excite(l, d + nob)
                                        if bstr is not None:
                                            idx = basis.index((refa,bstr))
                                            C[idx] += s1*s2*s3*s4*T1b[a,i]*T1b[b,j]*T2bb[c,d,k,l]/2.0

        # loop over T1a^4
        for i in range(noa):
            for j in range(noa):
                for k in range(noa):
                    for l in range(noa):
                        for a in range(nva):
                            for b in range(nva):
                                for c in range(nva):
                                    for d in range(nva):
                                        sa1, astr = refa.excite(i, a + noa)
                                        if astr is None:
                                            continue
                                        sa2, astr = astr.excite(j, b + noa)
                                        if astr is None:
                                            continue
                                        sa3, astr = astr.excite(k, c + noa)
                                        if astr is None:
                                            continue
                                        sa4, astr = astr.excite(l, d + noa)
                                        if astr is not None:
                                            idx = basis.index((astr,refb))
                                            C[idx] += sa1*sa2*sa3*sa4*T1a[a,i]*T1a[b,j]*T1a[c,k]*T1a[d,l]/24.0
        # loop over T1a^3T1b
        for i in range(noa):
            for j in range(noa):
                for k in range(noa):
                    for l in range(nob):
                        for a in range(nva):
                            for b in range(nva):
                                for c in range(nva):
                                    for d in range(nvb):
                                        sa1, astr = refa.excite(i, a + noa)
                                        if astr is None:
                                            continue
                                        sa2, astr = astr.excite(j, b + noa)
                                        if astr is None:
                                            continue
                                        sa3, astr = astr.excite(k, c + noa)
                                        sb1, bstr = refb.excite(l, d + nob)
                                        if bstr is not None and astr is not None:
                                            idx = basis.index((astr,bstr))
                                            C[idx] += sa1*sa2*sa3*sb1*T1a[a,i]*T1a[b,j]*T1a[c,k]*T1b[d,l]/6.0
        # loop over T1a^2T1b^2
        for i in range(noa):
            for j in range(noa):
                for k in range(nob):
                    for l in range(nob):
                        for a in range(nva):
                            for b in range(nva):
                                for c in range(nvb):
                                    for d in range(nvb):
                                        sa1, astr = refa.excite(i, a + noa)
                                        if astr is None:
                                            continue
                                        sa2, astr = astr.excite(j, b + noa)
                                        sb1, bstr = refb.excite(k, c + nob)
                                        if bstr is None:
                                            continue
                                        sb2, bstr = bstr.excite(l, d + nob)
                                        if bstr is not None and astr is not None:
                                            idx = basis.index((astr,bstr))
                                            C[idx] += sa1*sa2*sb1*sb2*T1a[a,i]*T1a[b,j]*T1b[c,k]*T1b[d,l]/4.0
        # loop over T1aT1b^3
        for i in range(noa):
            for j in range(nob):
                for k in range(nob):
                    for l in range(nob):
                        for a in range(nva):
                            for b in range(nvb):
                                for c in range(nvb):
                                    for d in range(nvb):
                                        sa, astr = refa.excite(i, a + noa)
                                        s1, bstr = refb.excite(j, b + nob)
                                        if bstr is None:
                                            continue
                                        s2, bstr = bstr.excite(k, c + nob)
                                        if bstr is None:
                                            continue
                                        s3, bstr = bstr.excite(l, d + nob)
                                        if bstr is not None and astr is not None:
                                            idx = basis.index((astr,bstr))
                                            C[idx] += sa*s1*s2*s3*T1a[a,i]*T1b[b,j]*T1b[c,k]*T1b[d,l]/6.0
        # loop over T1b^4
        for i in range(nob):
            for j in range(nob):
                for k in range(nob):
                    for l in range(nob):
                        for a in range(nvb):
                            for b in range(nvb):
                                for c in range(nvb):
                                    for d in range(nvb):
                                        s1, bstr = refb.excite(i, a + nob)
                                        if bstr is None:
                                            continue
                                        s2, bstr = bstr.excite(j, b + nob)
                                        if bstr is None:
                                            continue
                                        s3, bstr = bstr.excite(k, c + nob)
                                        if bstr is None:
                                            continue
                                        s4, bstr = bstr.excite(l, d + nob)
                                        if bstr is not None:
                                            idx = basis.index((refa,bstr))
                                            C[idx] += s1*s2*s3*s4*T1b[a,i]*T1b[b,j]*T1b[c,k]*T1b[d,l]/24.0
    if order >= 5:
        raise Exception("Order {} is not supported".format(order))
    return C


def gci_matrixel(bra, ket, h, I, const):
    ddd = diff(bra, ket)//2
    aa = [(ob - ok) for ob,ok in zip(bra.occ, ket.occ)]
    ao = [-1 if a < 0 else 0 for a in aa]
    av = [1 if a > 0 else 0 for a in aa]
    if ddd == 0:
        p = bra.occ.copy()
        Hel = numpy.einsum('ii,i->', h, p)
        Hel += 0.5*numpy.einsum('ijij,i,j->', I, p, p)
        Hel -= 0.5*numpy.einsum('ijji,i,j->', I, p, p)
        return Hel
    elif ddd == 1:
        o = numpy.nonzero(ao)[0][0]
        v = numpy.nonzero(av)[0][0]
        p = ket.occ.copy()
        b = bra.occ.copy()
        i1 = min(o, v)
        i2 = max(o, v)
        sign = 1.0 if b[i1+1:i2].sum() % 2 == 0 else -1.0
        Hel = 0.0
        Hel += h[v,o]
        Hel += numpy.einsum('ii,i->',I[v,:,o,:],p)
        Hel -= numpy.einsum('ii,i->',I[v,:,:,o],p)
        return sign*Hel
    elif ddd == 2:
        o = numpy.nonzero(ao)[0]
        v = numpy.nonzero(av)[0]
        o1, o2 = o
        v1, v2 = v
        b = bra.occ.copy()
        i1, i2 = min(o1, v1), max(o1, v1)
        j1, j2 = min(o2, v2), max(o2, v2)
        sign1 = 1.0 if b[i1+1:i2].sum() % 2 == 0 else -1.0
        sign2 = 1.0 if b[j1+1:j2].sum() % 2 == 0 else -1.0
        sign = -sign1*sign2 if (v2 < o1 or o2 < v1) else sign1*sign2
        # from Ha
        Hel = I[v1,v2,o1,o2] - I[v1,v2,o2,o1]
        return sign1*sign2*Hel
    else:
        return 0.0


def ci_matrixel(braa, brab, keta, ketb, ha, hb, Ia, Ib, Iabab, const):
    diffa = diff(braa, keta)//2
    diffb = diff(brab, ketb)//2
    aa = [(ob - ok) for ob,ok in zip(braa.occ, keta.occ)]
    bb = [(ob - ok) for ob,ok in zip(brab.occ, ketb.occ)]
    ao = [-1 if a < 0 else 0 for a in aa]
    av = [1 if a > 0 else 0 for a in aa]
    bo = [-1 if b < 0 else 0 for b in bb]
    bv = [1 if b > 0 else 0 for b in bb]
    if diffa == 0 and diffb == 0:
        # from Ha
        pa = braa.occ.copy()
        Ea = numpy.einsum('ii,i->', ha, pa)
        Ea += 0.5*numpy.einsum('ijij,i,j->', Ia, pa, pa)
        Ea -= 0.5*numpy.einsum('ijji,i,j->', Ia, pa, pa)

        # from Hb
        pb = brab.occ.copy()
        Eb = numpy.einsum('ii,i->', hb, pb)
        Eb += 0.5*numpy.einsum('ijij,i,j->', Ib, pb, pb)
        Eb -= 0.5*numpy.einsum('ijji,i,j->', Ib, pb, pb)

        # from Hab
        Eab = numpy.einsum('ijij,i,j', Iabab, pa, pb)
        Hel = Ea + Eb + Eab

        return Hel + const

    elif diffa == 1 and diffb == 0:
        o = numpy.nonzero(ao)[0][0]
        v = numpy.nonzero(av)[0][0]
        pa = keta.occ.copy()
        pb = ketb.occ.copy()
        ba = braa.occ.copy()
        i1 = min(o, v)
        i2 = max(o, v)
        sign = 1.0 if ba[i1+1:i2].sum() % 2 == 0 else -1.0
        Hel = 0.0
        # from Ha
        Hel += ha[v,o]
        Hel += numpy.einsum('ii,i->', Ia[v,:,o,:], pa)
        Hel -= numpy.einsum('ii,i->', Ia[v,:,:,o], pa)
        # from Hab
        Hel += numpy.einsum('ii,i->', Iabab[v,:,o,:], pb)
        return sign*Hel

    elif diffa == 0 and diffb == 1:
        o = numpy.nonzero(bo)[0][0]
        v = numpy.nonzero(bv)[0][0]
        pa = keta.occ.copy()
        pb = ketb.occ.copy()
        bb = brab.occ.copy()
        i1 = min(o, v)
        i2 = max(o, v)
        sign = 1.0 if bb[i1+1:i2].sum() % 2 == 0 else -1.0
        Hel = 0.0
        # from Hb
        Hel += hb[v,o]
        Hel += numpy.einsum('ii,i->', Ib[v,:,o,:], pb)
        Hel -= numpy.einsum('ii,i->', Ib[v,:,:,o], pb)
        # from Hab
        Hel += numpy.einsum('ii,i->', Iabab[:,v,:,o], pa)
        return sign*Hel

    elif diffa == 1 and diffb == 1:
        oa = numpy.nonzero(ao)[0][0]
        va = numpy.nonzero(av)[0][0]
        ob = numpy.nonzero(bo)[0][0]
        vb = numpy.nonzero(bv)[0][0]
        ba = braa.occ.copy()
        bb = brab.occ.copy()
        i1a = min(oa, va)
        i2a = max(oa, va)
        i1b = min(ob, vb)
        i2b = max(ob, vb)
        signa = 1.0 if ba[i1a+1:i2a].sum() % 2 == 0 else -1.0
        signb = 1.0 if bb[i1b+1:i2b].sum() % 2 == 0 else -1.0
        # from Hab
        Hel = Iabab[va,vb,oa,ob]
        return signa*signb*Hel

    elif diffa == 2 and diffb == 0:
        o = numpy.nonzero(ao)[0]
        v = numpy.nonzero(av)[0]
        o1, o2 = o
        v1, v2 = v
        ba = braa.occ.copy()
        i1, i2 = min(o1, v1), max(o1, v1)
        j1, j2 = min(o2, v2), max(o2, v2)
        sign1 = 1.0 if ba[i1+1:i2].sum() % 2 == 0 else -1.0
        sign2 = 1.0 if ba[j1+1:j2].sum() % 2 == 0 else -1.0
        sign = -sign1*sign2 if (v2 < o1 or o2 < v1) else sign1*sign2
        # from Ha
        Hel = Ia[v1,v2,o1,o2] - Ia[v1,v2,o2,o1]
        return sign*Hel

    elif diffa == 0 and diffb == 2:
        o = numpy.nonzero(bo)[0]
        v = numpy.nonzero(bv)[0]
        o1, o2 = o
        v1, v2 = v
        bb = brab.occ.copy()
        i1, i2 = min(o1, v1), max(o1, v1)
        j1, j2 = min(o2, v2), max(o2, v2)
        sign1 = 1.0 if bb[i1+1:i2].sum() % 2 == 0 else -1.0
        sign2 = 1.0 if bb[j1+1:j2].sum() % 2 == 0 else -1.0
        sign = -sign1*sign2 if (v2 < o1 or o2 < v1) else sign1*sign2
        # from Hb
        Hel = Ib[v1,v2,o1,o2] - Ib[v1,v2,o2,o1]
        return sign*Hel

    else:
        return 0.0


def s_on_vec(basis, vec, i, a):
    out = numpy.zeros(vec.shape)
    for ix,x in enumerate(vec):
        dd = basis[ix]
        s, new = dd.excite(i, a)
        if new is None:
            continue
        try:
            idx = basis.index(new)
            out[idx] += s*x
        except ValueError:
            pass
    return out


def sa_on_vec(basis, vec, i, a):
    out = numpy.zeros(vec.shape)
    for ix,x in enumerate(vec):
        ab, bb = basis[ix]
        sa, anew = ab.excite(i, a)
        if anew is None:
            continue
        try:
            idx = basis.index((anew,bb))
            out[idx] += sa*x
        except ValueError:
            pass
    return out


def sb_on_vec(basis, vec, i, a):
    out = numpy.zeros(vec.shape)
    for ix,x in enumerate(vec):
        ab, bb = basis[ix]
        sb, bnew = bb.excite(i, a)
        if bnew is None:
            continue
        try:
            idx = basis.index((ab,bnew))
            out[idx] += sb*x
        except ValueError:
            pass
    return out


def d_on_vec(basis, vec, i, j, a, b):
    out = numpy.zeros(vec.shape)
    for ix,x in enumerate(vec):
        dd = basis[ix]
        s1, temp = dd.excite(i, a)
        if temp is None:
            continue
        s2, new = temp.excite(j, b)
        if new is None:
            continue
        try:
            idx = basis.index(new)
            out[idx] += s1*s2*x
        except ValueError:
            pass
    return out


def da_on_vec(basis, vec, i, j, a, b):
    out = numpy.zeros(vec.shape)
    for ix,x in enumerate(vec):
        ab, bb = basis[ix]
        s1, atemp = ab.excite(i, a)
        if atemp is None:
            continue
        s2, anew = atemp.excite(j, b)
        if anew is None:
            continue
        try:
            idx = basis.index((anew,bb))
            out[idx] += s1*s2*x
        except ValueError:
            pass
    return out


def db_on_vec(basis, vec, i, j, a, b):
    out = numpy.zeros(vec.shape)
    for ix,x in enumerate(vec):
        ab, bb = basis[ix]
        s1, btemp = bb.excite(i, a)
        if btemp is None:
            continue
        s2, bnew = btemp.excite(j, b)
        if bnew is None:
            continue
        try:
            idx = basis.index((ab,bnew))
            out[idx] += s1*s2*x
        except ValueError:
            pass
    return out


def sasb_on_vec(basis, vec, i, j, a, b):
    out = numpy.zeros(vec.shape)
    for ix,x in enumerate(vec):
        ab, bb = basis[ix]
        sa, anew = ab.excite(i, a)
        if anew is None:
            continue
        sb, bnew = bb.excite(j, b)
        if bnew is None:
            continue
        try:
            idx = basis.index((anew,bnew))
            out[idx] += sa*sb*x
        except ValueError:
            pass
    return out


def H_on_vec(basis, vec, ha, hb, Ia, Ib, Iabab):
    out = numpy.zeros(vec.shape)
    for i,b in enumerate(basis):
        for j,k in enumerate(basis):
            out[i] += vec[j]*ci_matrixel(b[0], b[1], k[0], k[1], ha, hb, Ia, Ib, Iabab, 0.0)
    return out


def gH_on_vec(basis, vec, h, I,):
    out = numpy.zeros(vec.shape)
    for i,b in enumerate(basis):
        for j,k in enumerate(basis):
            out[i] += vec[j]*gci_matrixel(b, k, h, I, 0.0)
    return out

import numpy
import time
from pyscf import lib

einsum = lib.einsum
#einsum = einsum

def _S_S(T1, F, I, T1old, fac=1.0):
    # S[S]-A
    T1 += fac*einsum('ab,bi->ai',F.vv,T1old)
    # S[S]-B
    T1 -= fac*einsum('ji,aj->ai',F.oo,T1old)
    # S[S]-C
    T1 -= fac*einsum('ajbi,bj->ai',I.vovo,T1old)

def _S_D(T1, F, I, T2old, fac=1.0):
    # S[D]-A
    T1 += fac*einsum('jb,abij->ai',F.ov,T2old)
    # S[D]-B
    T1 += fac*0.5*einsum('ajbc,bcij->ai',I.vovv,T2old)
    # S[D]-C
    T1 -= fac*0.5*einsum('jkib,abjk->ai',I.ooov,T2old)

def _S_SS(T1, F, I, T1old, fac=1.0):
    # S[SS]-A
    T1 -= fac*einsum('jb,bi,aj->ai',F.ov,T1old,T1old)
    # S[SS]-B
    T1 -= fac*einsum('ajbc,bj,ci->ai',I.vovv,T1old,T1old)
    # S[SS]-C
    T1 += fac*einsum('jkib,bj,ak->ai',I.ooov,T1old,T1old)

def _S_SD(T1, F, I, T1old, T2old, fac=1.0):
    # S[SD]-A
    T1 -= fac*0.5*einsum('jkbc,bi,acjk->ai',I.oovv,T1old,T2old)
    # S[SD]-B
    T1 -= fac*0.5*einsum('jkbc,aj,bcik->ai',I.oovv,T1old,T2old)
    # S[SD]-C
    T1 += fac*einsum('jkbc,bj,caki->ai',I.oovv,T1old,T2old)

def _S_SSS(T1, F, I, T1old, fac=1.0):
    T1 += fac*einsum('jkbc,bi,cj,ak->ai',I.oovv,T1old,T1old,T1old)

def _D_S(T2, F, I, T1old, fac=1.0):
    # D[S]-A
    T2 += fac*einsum('abcj,ci->abij',I.vvvo,T1old)
    T2 -= fac*einsum('abci,cj->abij',I.vvvo,T1old)
    # D[S]-B
    T2 += fac*einsum('bkij,ak->abij',I.vooo,T1old)
    T2 -= fac*einsum('akij,bk->abij',I.vooo,T1old)

def _D_D(T2, F, I, T2old, fac=1.0):
    # D[D]-A
    T2 += fac*einsum('bc,acij->abij',F.vv,T2old)
    T2 -= fac*einsum('ac,bcij->abij',F.vv,T2old)
    # D[D]-B
    T2 -= fac*einsum('kj,abik->abij',F.oo,T2old)
    T2 += fac*einsum('ki,abjk->abij',F.oo,T2old)
    # D[D]-C
    T2 += fac*0.5*einsum('abcd,cdij->abij',I.vvvv,T2old)
    # D[D]-D
    T2 += fac*0.5*einsum('klij,abkl->abij',I.oooo,T2old)
    # D[D]-E
    T2 -= fac*einsum('bkcj,acik->abij',I.vovo,T2old)
    T2 += fac*einsum('akcj,bcik->abij',I.vovo,T2old)
    T2 += fac*einsum('bkci,acjk->abij',I.vovo,T2old)
    T2 -= fac*einsum('akci,bcjk->abij',I.vovo,T2old)

def _D_SS(T2, F, I, T1old, fac=1.0):
    # D[SS]-A
    T2 += fac*0.5*einsum('abcd,ci,dj->abij',I.vvvv,T1old,T1old)
    T2 -= fac*0.5*einsum('abcd,cj,di->abij',I.vvvv,T1old,T1old)
    # D[SS]-B
    T2 += fac*0.5*einsum('klij,ak,bl->abij',I.oooo,T1old,T1old)
    T2 -= fac*0.5*einsum('klij,bk,al->abij',I.oooo,T1old,T1old)
    # D[SS]-C
    T2 -= fac*einsum('akcj,ci,bk->abij',I.vovo,T1old,T1old)
    T2 += fac*einsum('bkcj,ci,ak->abij',I.vovo,T1old,T1old)
    T2 += fac*einsum('akci,cj,bk->abij',I.vovo,T1old,T1old)
    T2 -= fac*einsum('bkci,cj,ak->abij',I.vovo,T1old,T1old)

def _D_SD(T2, F, I, T1old, T2old, fac=1.0):
    # D[SD]-A
    T2 -= fac*einsum('kc,ci,abkj->abij',F.ov,T1old,T2old)
    T2 += fac*einsum('kc,cj,abki->abij',F.ov,T1old,T2old)
    # D[SD]-B
    T2 -= fac*einsum('kc,ak,cbij->abij',F.ov,T1old,T2old)
    T2 += fac*einsum('kc,bk,caij->abij',F.ov,T1old,T2old)
    # D[SD]-C
    T2 -= fac*einsum('akcd,ck,dbij->abij',I.vovv,T1old,T2old)
    T2 += fac*einsum('bkcd,ck,daij->abij',I.vovv,T1old,T2old)
    # D[SD]-D
    T2 += fac*einsum('klic,ck,ablj->abij',I.ooov,T1old,T2old)
    T2 -= fac*einsum('kljc,ck,abli->abij',I.ooov,T1old,T2old)
    # D[SD]-E
    T2 += fac*einsum('akcd,ci,dbkj->abij',I.vovv,T1old,T2old)
    T2 -= fac*einsum('bkcd,ci,dakj->abij',I.vovv,T1old,T2old)
    T2 -= fac*einsum('akcd,cj,dbki->abij',I.vovv,T1old,T2old)
    T2 += fac*einsum('bkcd,cj,daki->abij',I.vovv,T1old,T2old)
    # D[SD]-F
    T2 -= fac*einsum('klic,ak,cblj->abij',I.ooov,T1old,T2old)
    T2 += fac*einsum('klic,bk,calj->abij',I.ooov,T1old,T2old)
    T2 += fac*einsum('kljc,ak,cbli->abij',I.ooov,T1old,T2old)
    T2 -= fac*einsum('kljc,bk,cali->abij',I.ooov,T1old,T2old)
    # D[SD]-G
    T2 -= fac*0.5*einsum('kljc,ci,abkl->abij',I.ooov,T1old,T2old)
    T2 += fac*0.5*einsum('klic,cj,abkl->abij',I.ooov,T1old,T2old)
    # D[SD]-H
    T2 += fac*0.5*einsum('bkcd,ak,cdij->abij',I.vovv,T1old,T2old)
    T2 -= fac*0.5*einsum('akcd,bk,cdij->abij',I.vovv,T1old,T2old)

def _D_DD(T2, F, I, T2old, fac=1.0):
    # D[DD]-A
    T2 += fac*0.25*einsum('klcd,cdij,abkl->abij',I.oovv,T2old,T2old)
    # D[DD]-B
    T2 += fac*einsum('klcd,acik,dblj->abij',I.oovv,T2old,T2old)
    T2 -= fac*einsum('klcd,bcik,dalj->abij',I.oovv,T2old,T2old)
    # D[DD]-C
    T2 -= fac*0.5*einsum('klcd,cakl,dbij->abij',I.oovv,T2old,T2old)
    T2 += fac*0.5*einsum('klcd,cbkl,daij->abij',I.oovv,T2old,T2old)
    # D[DD]-D
    T2 -= fac*0.5*einsum('klcd,cdki,ablj->abij',I.oovv,T2old,T2old)
    T2 += fac*0.5*einsum('klcd,cdkj,abli->abij',I.oovv,T2old,T2old)

def _D_SSS(T2, F, I, T1old, fac=1.0):
    # D[SSS]-A
    T2 += fac*einsum('bkcd,ci,ak,dj->abij',I.vovv,T1old,T1old,T1old)
    T2 -= fac*einsum('akcd,ci,bk,dj->abij',I.vovv,T1old,T1old,T1old)
    # D[SSS]-B
    T2 -= fac*einsum('kljc,ci,ak,bl->abij',I.ooov,T1old,T1old,T1old)
    T2 += fac*einsum('klic,cj,ak,bl->abij',I.ooov,T1old,T1old,T1old)

def _D_SSD(T2, F, I, T1old, T2old, fac=1.0):
    # D[SSD]-A
    T2 += fac*0.25*einsum('klcd,ci,dj,abkl->abij',I.oovv,T1old,T1old,T2old)
    T2 -= fac*0.25*einsum('klcd,cj,di,abkl->abij',I.oovv,T1old,T1old,T2old)
    # D[SSD]-B
    T2 += fac*0.25*einsum('klcd,ak,bl,cdij->abij',I.oovv,T1old,T1old,T2old)
    T2 -= fac*0.25*einsum('klcd,bk,al,cdij->abij',I.oovv,T1old,T1old,T2old)
    # D[SSD]-C
    T2 -= fac*einsum('klcd,ci,ak,dblj->abij',I.oovv,T1old,T1old,T2old)
    T2 += fac*einsum('klcd,ci,bk,dalj->abij',I.oovv,T1old,T1old,T2old)
    T2 += fac*einsum('klcd,cj,ak,dbli->abij',I.oovv,T1old,T1old,T2old)
    T2 -= fac*einsum('klcd,cj,bk,dali->abij',I.oovv,T1old,T1old,T2old)
    # D[SSD]-D
    T2 -= fac*einsum('klcd,ck,di,ablj->abij',I.oovv,T1old,T1old,T2old)
    T2 += fac*einsum('klcd,ck,dj,abli->abij',I.oovv,T1old,T1old,T2old)
    # D[SSD]-E
    T2 -= fac*einsum('klcd,ck,al,dbij->abij',I.oovv,T1old,T1old,T2old)
    T2 += fac*einsum('klcd,ck,bl,daij->abij',I.oovv,T1old,T1old,T2old)

def _D_SSSS(T2, F, I, T1old, fac=1.0):
    T2 += fac*einsum('klcd,ci,ak,bl,dj->abij',I.oovv,T1old,T1old,T1old,T1old)

def _Stanton(T1, T2, F, I, T1old, T2old, fac=1.0):

    T2A = T2old.copy()
    T2A += 0.5*einsum('ai,bj->abij',T1old,T1old)
    T2A -= 0.5*einsum('bi,aj->abij',T1old,T1old)

    Fvv = F.vv.copy()
    Fvv -= 0.5*einsum('jb,aj->ab',F.ov,T1old)
    Fvv -= einsum('ajcb,cj->ab',I.vovv,T1old)
    Fvv -= 0.5*einsum('jkbc,acjk->ab',I.oovv,T2A)

    Foo = F.oo.copy()
    Foo += 0.5*einsum('jb,bi->ji',F.ov,T1old)
    Foo += einsum('jkib,bk->ji',I.ooov,T1old)
    Foo += 0.5*einsum('jkbc,bcik->ji',I.oovv,T2A)
    T2A = None

    Fov = F.ov.copy()
    Fov += einsum('jkbc,ck->jb',I.oovv,T1old)

    T1 += fac*einsum('ab,bi->ai',Fvv,T1old)
    T1 -= fac*einsum('ji,aj->ai',Foo,T1old)
    T1 += fac*einsum('jb,abij->ai',Fov,T2old)
    T1 -= fac*einsum('ajbi,bj->ai',I.vovo,T1old)
    T1 += fac*0.5*einsum('ajbc,bcij->ai',I.vovv,T2old)
    T1 -= fac*0.5*einsum('jkib,abjk->ai',I.ooov,T2old)
    
    T2B = T2old.copy()
    T2B += einsum('ai,bj->abij',T1old,T1old)
    T2B -= einsum('bi,aj->abij',T1old,T1old)

    Woooo = I.oooo.copy()
    Woooo += einsum('klic,cj->klij',I.ooov,T1old)
    Woooo -= einsum('kljc,ci->klij',I.ooov,T1old)
    Woooo += 0.25*einsum('klcd,cdij->klij',I.oovv,T2B)
    T2 += fac*0.5*einsum('klij,abkl->abij',Woooo,T2B)
    Woooo = None

    Wvvvv = I.vvvv.copy()
    Wvvvv -= einsum('akcd,bk->abcd',I.vovv,T1old)
    Wvvvv += einsum('bkcd,ak->abcd',I.vovv,T1old)
    Wvvvv += 0.25*einsum('klcd,abkl->abcd',I.oovv,T2B)
    T2 += fac*0.5*einsum('abcd,cdij->abij',Wvvvv,T2B)
    T2B = None
    Wvvvv = None

    Wovvo = -I.vovo.transpose((1,0,2,3))
    Wovvo -= einsum('bkcd,dj->kbcj',I.vovv,T1old)
    Wovvo += einsum('kljc,bl->kbcj',I.ooov,T1old)
    temp = 0.5*T2old + einsum('dj,bl->dbjl',T1old,T1old)
    Wovvo -= einsum('klcd,dbjl->kbcj',I.oovv,temp)
    temp = einsum('kbcj,acik->abij',Wovvo,T2old)
    temp += einsum('bkcj,ci,ak->abij',I.vovo,T1old,T1old)
    T2 += fac*temp
    T2 -= fac*temp.transpose((0,1,3,2))
    T2 -= fac*temp.transpose((1,0,2,3))
    T2 += fac*temp.transpose((1,0,3,2))
    temp = None
    Wovvo = None

    Ftemp = Fvv - 0.5*einsum('jb,aj->ab',Fov,T1old)
    temp_ab = einsum('bc,acij->abij',Ftemp,T2old)
    temp_ab += einsum('bkij,ak->abij',I.vooo,T1old)
    T2 += fac*temp_ab
    T2 -= fac*temp_ab.transpose((1,0,2,3))
    temp_ab = None

    Ftemp = Foo + 0.5*einsum('jb,bi->ji',Fov,T1old)
    temp_ij = -einsum('kj,abik->abij',Ftemp,T2old)
    temp_ij += einsum('abcj,ci->abij',I.vvvo,T1old)
    T2 += fac*temp_ij
    T2 -= fac*temp_ij.transpose((0,1,3,2))
    temp_ij = None

def _u_Stanton(T1a, T1b, T2aa, T2ab, T2bb, Faa, Fbb, Ia, Ib, Iabab, T1old, T2old, fac=1.0):

    # unpack
    T1aold,T1bold = T1old
    T2aaold,T2abold,T2bbold = T2old

    nva,noa = T1a.shape
    nvb,nob = T1b.shape
    no = noa + nob
    nv = nva + nvb

    T2Aaa = T2aaold.copy()
    T2Aaa += 0.5*einsum('ai,bj->abij',T1aold,T1aold)
    T2Aaa -= 0.5*einsum('bi,aj->abij',T1aold,T1aold)
    T2Abb = T2bbold.copy()
    T2Abb += 0.5*einsum('ai,bj->abij',T1bold,T1bold)
    T2Abb -= 0.5*einsum('bi,aj->abij',T1bold,T1bold)
    T2Aab = T2abold.copy()
    T2Aab += 0.5*einsum('ai,bj->abij',T1aold,T1bold)

    Fvva = Faa.vv.copy()
    Fvva -= 0.5*einsum('jb,aj->ab',Faa.ov,T1aold)
    Fvva -= einsum('ajcb,cj->ab',Ia.vovv,T1aold)
    Fvva += einsum('ajbc,cj->ab',Iabab.vovv,T1bold)
    Fvva -= 0.5*einsum('jkbc,acjk->ab',Ia.oovv,T2Aaa)
    Fvva -= einsum('jkbc,acjk->ab',Iabab.oovv,T2Aab)

    Fvvb = Fbb.vv.copy()
    Fvvb -= 0.5*einsum('jb,aj->ab',Fbb.ov,T1bold)
    Fvvb -= einsum('ajcb,cj->ab',Ib.vovv,T1bold)
    Fvvb += einsum('jacb,cj->ab',Iabab.ovvv,T1aold)
    Fvvb -= 0.5*einsum('jkbc,acjk->ab',Ib.oovv,T2Abb)
    Fvvb -= einsum('kjcb,cakj->ab',Iabab.oovv,T2Aab)

    Fooa = Faa.oo.copy()
    Fooa += 0.5*einsum('jb,bi->ji',Faa.ov,T1aold)
    Fooa += einsum('jkib,bk->ji',Ia.ooov,T1aold)
    Fooa += einsum('jkib,bk->ji',Iabab.ooov,T1bold)
    Fooa += 0.5*einsum('jkbc,bcik->ji',Ia.oovv,T2Aaa)
    Fooa += einsum('jkbc,bcik->ji',Iabab.oovv,T2Aab)

    Foob = Fbb.oo.copy()
    Foob += 0.5*einsum('jb,bi->ji',Fbb.ov,T1bold)
    Foob += einsum('jkib,bk->ji',Ib.ooov,T1bold)
    Foob += einsum('kjbi,bk->ji',Iabab.oovo,T1aold)
    Foob += 0.5*einsum('jkbc,bcik->ji',Ib.oovv,T2Abb)
    Foob += einsum('kjcb,cbki->ji',Iabab.oovv,T2Aab)
    T2Aaa = None
    T2Aab = None
    T2Abb = None

    Fova = Faa.ov.copy()
    Fova += einsum('jkbc,ck->jb',Ia.oovv,T1aold)
    Fova += einsum('jkbc,ck->jb',Iabab.oovv,T1bold)

    Fovb = Fbb.ov.copy()
    Fovb += einsum('jkbc,ck->jb',Ib.oovv,T1bold)
    Fovb += einsum('kjcb,ck->jb',Iabab.oovv,T1aold)

    T1a += fac*einsum('ab,bi->ai',Fvva,T1aold)
    T1b += fac*einsum('ab,bi->ai',Fvvb,T1bold)
    T1a -= fac*einsum('ji,aj->ai',Fooa,T1aold)
    T1b -= fac*einsum('ji,aj->ai',Foob,T1bold)
    T1a += fac*einsum('jb,abij->ai',Fova,T2aaold)
    T1a += fac*einsum('jb,abij->ai',Fovb,T2abold)
    T1b += fac*einsum('jb,abij->ai',Fovb,T2bbold)
    T1b += fac*einsum('jb,baji->ai',Fova,T2abold)

    T1a -= fac*einsum('ajbi,bj->ai',Ia.vovo,T1aold)
    T1a += fac*einsum('ajib,bj->ai',Iabab.voov,T1bold)
    T1b -= fac*einsum('ajbi,bj->ai',Ib.vovo,T1bold)
    T1b += fac*einsum('jabi,bj->ai',Iabab.ovvo,T1aold)

    T1a += fac*0.5*einsum('ajbc,bcij->ai',Ia.vovv,T2aaold)
    T1a += 2*fac*0.5*einsum('ajbc,bcij->ai',Iabab.vovv,T2abold)
    T1b += fac*0.5*einsum('ajbc,bcij->ai',Ib.vovv,T2bbold)
    T1b += 2*fac*0.5*einsum('jacb,cbji->ai',Iabab.ovvv,T2abold)

    T1a -= fac*0.5*einsum('jkib,abjk->ai',Ia.ooov,T2aaold)
    T1a -= fac*einsum('jkib,abjk->ai',Iabab.ooov,T2abold)
    T1b -= fac*0.5*einsum('jkib,abjk->ai',Ib.ooov,T2bbold)
    T1b -= fac*einsum('kjbi,bakj->ai',Iabab.oovo,T2abold)

    T2Baa = T2aaold.copy()
    T2Baa += einsum('ai,bj->abij',T1aold,T1aold)
    T2Baa -= einsum('bi,aj->abij',T1aold,T1aold)
    T2Bbb = T2bbold.copy()
    T2Bbb += einsum('ai,bj->abij',T1bold,T1bold)
    T2Bbb -= einsum('bi,aj->abij',T1bold,T1bold)
    T2Bab = T2abold.copy()
    T2Bab += einsum('ai,bj->abij',T1aold,T1bold)

    Wo_aaaa = Ia.oooo.copy()
    Wo_aaaa += einsum('klic,cj->klij',Ia.ooov,T1aold)
    Wo_aaaa -= einsum('kljc,ci->klij',Ia.ooov,T1aold)
    Wo_aaaa += 0.25*einsum('klcd,cdij->klij',Ia.oovv,T2Baa)
    Wo_bbbb = Ib.oooo.copy()
    Wo_bbbb += einsum('klic,cj->klij',Ib.ooov,T1bold)
    Wo_bbbb -= einsum('kljc,ci->klij',Ib.ooov,T1bold)
    Wo_bbbb += 0.25*einsum('klcd,cdij->klij',Ib.oovv,T2Bbb)
    Wo_abab = Iabab.oooo.copy()
    Wo_abab += einsum('klic,cj->klij',Iabab.ooov,T1bold)
    Wo_abab += einsum('klcj,ci->klij',Iabab.oovo,T1aold)
    Wo_abab += 0.5*einsum('klcd,cdij->klij',Iabab.oovv,T2Bab)
    T2aa += fac*0.5*einsum('klij,abkl->abij',Wo_aaaa,T2Baa)
    T2bb += fac*0.5*einsum('klij,abkl->abij',Wo_bbbb,T2Bbb)
    T2ab += fac*einsum('klij,abkl->abij',Wo_abab,T2Bab)
    Wo_aaaa = None
    Wo_bbbb = None
    Wo_abab = None

    Wv_aaaa = Ia.vvvv.copy()
    Wv_aaaa -= einsum('akcd,bk->abcd',Ia.vovv,T1aold)
    Wv_aaaa += einsum('bkcd,ak->abcd',Ia.vovv,T1aold)
    Wv_aaaa += 0.25*einsum('klcd,abkl->abcd',Ia.oovv,T2Baa)
    Wv_bbbb = Ib.vvvv.copy()
    Wv_bbbb -= einsum('akcd,bk->abcd',Ib.vovv,T1bold)
    Wv_bbbb += einsum('bkcd,ak->abcd',Ib.vovv,T1bold)
    Wv_bbbb += 0.25*einsum('klcd,abkl->abcd',Ib.oovv,T2Bbb)
    Wv_abab = Iabab.vvvv.copy()
    Wv_abab -= einsum('akcd,bk->abcd',Iabab.vovv,T1bold)
    Wv_abab -= einsum('kbcd,ak->abcd',Iabab.ovvv,T1aold)
    Wv_abab += 0.5*einsum('klcd,abkl->abcd',Iabab.oovv,T2Bab)
    T2aa += fac*0.5*einsum('abcd,cdij->abij',Wv_aaaa,T2Baa)
    T2bb += fac*0.5*einsum('abcd,cdij->abij',Wv_bbbb,T2Bbb)
    T2ab += fac*einsum('abcd,cdij->abij',Wv_abab,T2Bab)
    T2Baa = None
    T2Bab = None
    T2Bbb = None
    Wv_aaaa = None
    Wv_bbbb = None
    Wv_abab = None

    aatemp = einsum('bkcj,ci,ak->abij',Ia.vovo,T1aold,T1aold)
    bbtemp = einsum('bkcj,ci,ak->abij',Ib.vovo,T1bold,T1bold)
    abtemp = -einsum('kbcj,ci,ak->abij',Iabab.ovvo,T1aold,T1aold)
    abtemp += -einsum('akcj,ci,bk->abij',Iabab.vovo,T1aold,T1bold)
    abtemp += -einsum('kbic,cj,ak->abij',Iabab.ovov,T1bold,T1aold)
    abtemp += -einsum('akic,cj,bk->abij',Iabab.voov,T1bold,T1bold)
    aatemp = aatemp + aatemp.transpose((1,0,3,2)) - aatemp.transpose((1,0,2,3)) - aatemp.transpose((0,1,3,2))
    bbtemp = bbtemp + bbtemp.transpose((1,0,3,2)) - bbtemp.transpose((1,0,2,3)) - bbtemp.transpose((0,1,3,2))
    T2aa += fac*aatemp
    T2bb += fac*bbtemp
    T2ab += fac*abtemp
    aatemp = None
    abtemp = None
    bbtemp = None

    TTTaa = 0.5*T2aaold + einsum('dj,bl->dbjl',T1aold,T1aold)
    TTTbb = 0.5*T2bbold + einsum('dj,bl->dbjl',T1bold,T1bold)
    TTTab = 0.5*T2abold + einsum('dj,bl->dbjl',T1aold,T1bold)

    W_ovvo = -Ia.vovo.transpose((1,0,2,3)).copy()
    W_ovvo -= einsum('bkcd,dj->kbcj',Ia.vovv,T1aold)
    W_ovvo += einsum('kljc,bl->kbcj',Ia.ooov,T1aold)
    W_ovvo -= einsum('klcd,dbjl->kbcj',Ia.oovv,TTTaa)
    W_ovvo += 0.5*einsum('klcd,bdjl->kbcj',Iabab.oovv,T2abold)

    W_OVVO = -Ib.vovo.transpose((1,0,2,3)).copy()
    W_OVVO -= einsum('bkcd,dj->kbcj',Ib.vovv,T1bold)
    W_OVVO += einsum('kljc,bl->kbcj',Ib.ooov,T1bold)
    W_OVVO -= einsum('klcd,dbjl->kbcj',Ib.oovv,TTTbb)
    W_OVVO += 0.5*einsum('lkdc,dblj->kbcj',Iabab.oovv,T2abold)

    W_oVvO = Iabab.ovvo.copy()
    W_oVvO += einsum('kbcd,dj->kbcj',Iabab.ovvv,T1bold)
    W_oVvO -= einsum('klcj,bl->kbcj',Iabab.oovo,T1bold)
    W_oVvO -= einsum('klcd,dbjl->kbcj',Iabab.oovv,TTTbb)
    W_oVvO += 0.5*einsum('klcd,dblj->kbcj',Ia.oovv,T2abold)

    W_OvVo = Iabab.voov.transpose((1,0,3,2)).copy()
    W_OvVo += einsum('bkdc,dj->kbcj',Iabab.vovv,T1aold)
    W_OvVo -= einsum('lkjc,bl->kbcj',Iabab.ooov,T1aold)
    W_OvVo -= einsum('lkdc,dbjl->kbcj',Iabab.oovv,TTTaa)
    W_OvVo += 0.5*einsum('klcd,bdjl->kbcj',Ib.oovv,T2abold)

    W_oVVo = -Iabab.ovov.transpose((0,1,3,2)).copy()
    W_oVVo -= einsum('kbdc,dj->kbcj',Iabab.ovvv,T1aold)
    W_oVVo += einsum('kljc,bl->kbcj',Iabab.ooov,T1bold)
    W_oVVo += einsum('kldc,dbjl->kbcj',Iabab.oovv,TTTab)

    W_OvvO = -Iabab.vovo.transpose((1,0,2,3)).copy()
    W_OvvO -= einsum('bkcd,dj->kbcj',Iabab.vovv,T1bold)
    W_OvvO += einsum('lkcj,bl->kbcj',Iabab.oovo,T1aold)
    W_OvvO += einsum('lkcd,bdlj->kbcj',Iabab.oovv,TTTab)

    ATaa = einsum('kbcj,acik->abij',W_ovvo,T2aaold)
    ATaa += einsum('kbcj,acik->abij',W_OvVo,T2abold)
    ATbb = einsum('kbcj,acik->abij',W_OVVO,T2bbold)
    ATbb += einsum('kbcj,caki->abij',W_oVvO,T2abold)
    ATaa = ATaa - ATaa.transpose((0,1,3,2)) - ATaa.transpose((1,0,2,3)) + ATaa.transpose((1,0,3,2))
    ATbb = ATbb - ATbb.transpose((0,1,3,2)) - ATbb.transpose((1,0,2,3)) + ATbb.transpose((1,0,3,2))

    ATab = einsum('kbcj,acik->abij',W_oVvO,T2aaold)
    ATab += einsum('kbcj,acik->abij',W_OVVO,T2abold)
    ATab += einsum('kacj,cbik->abij',W_OvvO,T2abold)
    ATab += einsum('kbci,ackj->abij',W_oVVo,T2abold)
    ATab += einsum('kaci,cbkj->abij',W_ovvo,T2abold)
    ATab += einsum('kaci,cbkj->abij',W_OvVo,T2bbold)
    T2aa += fac*ATaa
    T2ab += fac*ATab
    T2bb += fac*ATbb

    W_ovvo = None
    W_OVVO = None
    W_oVvO = None
    W_OvVo = None
    W_oVVo = None
    W_OvvO = None

    Fvva -= 0.5*einsum('jb,aj->ab',Fova,T1aold)
    Fvvb -= 0.5*einsum('jb,aj->ab',Fovb,T1bold)
    ATaa = einsum('bc,acij->abij',Fvva,T2aaold)
    ATaa += einsum('bkij,ak->abij',Ia.vooo,T1aold)
    ATaa -= ATaa.transpose((1,0,2,3))
    ATbb = einsum('bc,acij->abij',Fvvb,T2bbold)
    ATbb += einsum('bkij,ak->abij',Ib.vooo,T1bold)
    ATbb -= ATbb.transpose((1,0,2,3))
    ATab = einsum('bc,acij->abij',Fvvb,T2abold)
    ATab += einsum('ac,cbij->abij',Fvva,T2abold)
    ATab -= einsum('kbij,ak->abij',Iabab.ovoo,T1aold)
    ATab -= einsum('akij,bk->abij',Iabab.vooo,T1bold)
    T2aa += fac*ATaa
    T2ab += fac*ATab
    T2bb += fac*ATbb

    Fooa += 0.5*einsum('jb,bi->ji',Fova,T1aold)
    Foob += 0.5*einsum('jb,bi->ji',Fovb,T1bold)
    ATaa = einsum('kj,abik->abij',Fooa,T2aaold)
    ATaa -= einsum('abcj,ci->abij',Ia.vvvo,T1aold)
    ATaa -= ATaa.transpose((0,1,3,2))
    ATbb = einsum('kj,abik->abij',Foob,T2bbold)
    ATbb -= einsum('abcj,ci->abij',Ib.vvvo,T1bold)
    ATbb -= ATbb.transpose((0,1,3,2))
    ATab = einsum('kj,abik->abij',Foob,T2abold)
    ATab += einsum('ki,abkj->abij',Fooa,T2abold)
    ATab -= einsum('abcj,ci->abij',Iabab.vvvo,T1aold)
    ATab -= einsum('abic,cj->abij',Iabab.vvov,T1bold)
    T2aa -= fac*ATaa
    T2ab -= fac*ATab
    T2bb -= fac*ATbb

def lccd_simple(F, I, T2old):
    """Linearized coupled cluster doubles (LCCD) iteration."""
    T2 = I.vvoo.copy()
    
    _D_D(T2, F, I, T2old)

    return T2

def lccsd_simple(F, I, T1old, T2old):
    """Linearized coupled cluster singles and doubles (LCCSD) iteration."""
    T1 = F.vo.copy()

    _S_S(T1, F, I, T1old)
    _S_D(T1, F, I, T2old)

    T2 = I.vvoo.copy()
    
    _D_S(T2, F, I, T1old)
    _D_D(T2, F, I, T2old)

    return T1,T2

def ccd_simple(F, I, T2old):
    """Coupled cluster doubles (CCD) iteration."""
    T2 = I.vvoo.copy()
    
    _D_D(T2, F, I, T2old)
    _D_DD(T2, F, I, T2old)

    return T2

def ccsd_simple(F, I, T1old, T2old):
    """Coupled cluster singles and doubles (CCSD) iteration."""
    T1 = F.vo.copy()

    _S_S(T1, F, I, T1old)
    _S_D(T1, F, I, T2old)
    _S_SS(T1, F, I, T1old)
    _S_SD(T1, F, I, T1old, T2old)
    _S_SSS(T1, F, I, T1old)

    T2 = I.vvoo.copy()
    
    _D_S(T2, F, I, T1old)
    _D_D(T2, F, I, T2old)
    _D_SS(T2, F, I, T1old)
    _D_SD(T2, F, I, T1old, T2old)
    _D_DD(T2, F, I, T2old)
    _D_SSS(T2, F, I, T1old)
    _D_SSD(T2, F, I, T1old, T2old)
    _D_SSSS(T2, F, I, T1old)

    return T1,T2

def ccsd_stanton(F, I, T1old, T2old):
    """Coupled cluster singles and doubles (CCSD) iteration
    using Stanton-Gauss intermediates.
    """
    T1 = F.vo.copy()
    T2 = I.vvoo.copy()

    _Stanton(T1, T2, F, I, T1old, T2old)

    return T1,T2

def uccsd_stanton(Faa, Fbb, Ia, Ib, I_abab, T1old, T2old):
    T1a = Faa.vo.copy()
    T1b = Fbb.vo.copy()
    T2aa = Ia.vvoo.copy()
    T2ab = I_abab.vvoo.copy()
    T2bb = Ib.vvoo.copy()

    _u_Stanton(T1a,T1b,T2aa,T2ab,T2bb,Faa,Fbb,Ia,Ib,I_abab,T1old,T2old)

    return (T1a,T1b),(T2aa,T2ab,T2bb)

def _LS_TS(L1, I, T1old, fac=1.0):
    L1 += fac*einsum('jiba,bj->ia',I.oovv,T1old)

def _u_LS_TS(L1a, L1b, Ia, Ib, Iabab, T1aold, T1bold, fac=1.0):
    L1a += fac*einsum('jiba,bj->ia',Ia.oovv,T1aold)
    L1a += fac*einsum('ijab,bj->ia',Iabab.oovv,T1bold)
    L1b += fac*einsum('jiba,bj->ia',Ib.oovv,T1bold)
    L1b += fac*einsum('jiba,bj->ia',Iabab.oovv,T1aold)

def _LS_LS(L1, F, I, L1old, fac=1.0):
    # A
    L1 += fac*einsum('ib,ba->ia',L1old,F.vv)
    # B
    L1 -= fac*einsum('ja,ij->ia',L1old,F.oo)
    # C
    #L1 += fac*einsum('jb,bija->ia',L1old,I.voov)
    L1 -= fac*einsum('jb,biaj->ia',L1old,I.vovo)

def _LS_LSTS(L1, F, I, L1old, T1old, fac=1.0):
    # A
    L1 -= fac*einsum('ja,ib,bj->ia',L1old,F.ov,T1old)
    # B
    L1 -= fac*einsum('ib,ja,bj->ia',L1old,F.ov,T1old)
    # C
    L1 += fac*einsum('ic,cjab,bj->ia',L1old,I.vovv,T1old)
    # D
    L1 -= fac*einsum('ka,ijkb,bj->ia',L1old,I.ooov,T1old)
    # E
    L1 += fac*einsum('jc,ciba,bj->ia',L1old,I.vovv,T1old)
    # F
    L1 -= fac*einsum('kb,jika,bj->ia',L1old,I.ooov,T1old)

def _LS_LSTD(L1, I, L1old, T2old, fac=1.0):
    # A
    L1 -= 0.5*fac*einsum('ja,ikbc,bcjk->ia',L1old,I.oovv,T2old)
    # B
    L1 -= 0.5*fac*einsum('ib,jkac,bcjk->ia',L1old,I.oovv,T2old)
    # C
    L1 += fac*einsum('jb,kica,bcjk->ia',L1old,I.oovv,T2old)

def _LS_LSTSS(L1, I, L1old, T1old, fac=1.0):
    # A
    L1 -= fac*einsum('ja,ikbc,bj,ck->ia',L1old,I.oovv,T1old,T1old)
    # B
    L1 -= fac*einsum('ib,jkac,bj,ck->ia',L1old,I.oovv,T1old,T1old)
    # C
    L1 -= fac*einsum('jc,kiba,bj,ck->ia',L1old,I.oovv,T1old,T1old)

def _LS_LD(L1, F, I, L2old, fac=1.0):
    # A
    L1 += 0.5*fac*einsum('ijcb,cbaj->ia',L2old,I.vvvo)
    # B
    #L1 -= 0.5*fac*einsum('kjab,ibkj->ia',L2old,I.ovoo)
    L1 += 0.5*fac*einsum('kjab,bikj->ia',L2old,I.vooo)

def _LS_LDTS(L1, F, I, L2old, T1old, fac=1.0):
    # A
    #L1 -= fac*einsum('jkac,icbk,bj->ia',L2old,I.vovo,T1old)
    L1 += fac*einsum('jkac,cibk,bj->ia',L2old,I.vovo,T1old)
    # B
    #L1 -= fac*einsum('ikbc,jcak,bj->ia',L2old,I.vovo,T1old)
    L1 += fac*einsum('ikbc,cjak,bj->ia',L2old,I.vovo,T1old)
    # C
    L1 += 0.5*fac*einsum('ijcd,cdab,bj->ia',L2old,I.vvvv,T1old)
    # D
    L1 += 0.5*fac*einsum('klab,ijkl,bj->ia',L2old,I.oooo,T1old)

def _LS_LDTD(L1, F, I, L2old, T2old, fac=1.0):
    # A
    L1 -= 0.5*fac*einsum('jkba,ic,bcjk->ia',L2old,F.ov,T2old)
    # B
    L1 -= 0.5*fac*einsum('jibc,ka,bcjk->ia',L2old,F.ov,T2old)
    # C
    L1 += 0.5*fac*einsum('jkbd,dica,bcjk->ia',L2old,I.vovv,T2old)
    # D
    L1 -= 0.5*fac*einsum('jlbc,kila,bcjk->ia',L2old,I.ooov,T2old)
    # E
    #L1 += fac*einsum('jibd,kdca,bcjk->ia',L2old,I.ovvv,T2old)
    L1 -= fac*einsum('jibd,dkca,bcjk->ia',L2old,I.vovv,T2old)
    # F
    #L1 -= fac*einsum('jlba,kicl,bcjk->ia',L2old,I.oovo,T2old)
    L1 += fac*einsum('jlba,kilc,bcjk->ia',L2old,I.ooov,T2old)
    # G
    #L1 -= 0.25*fac*einsum('jkad,idbc,bcjk->ia',L2old,I.ovvv,T2old)
    L1 += 0.25*fac*einsum('jkad,dibc,bcjk->ia',L2old,I.vovv,T2old)
    # H
    #L1 += 0.25*fac*einsum('ilbc,jkal,bcjk->ia',L2old,I.oovo,T2old)
    L1 -= 0.25*fac*einsum('ilbc,jkla,bcjk->ia',L2old,I.ooov,T2old)

def _LS_LDTSS(L1, F, I, L2old, T1old, fac=1.0):
    # A
    L1 -= fac*einsum('ikdb,djac,bj,ck->ia',L2old,I.vovv,T1old,T1old)
    # B
    L1 += fac*einsum('lkab,ijlc,bj,ck->ia',L2old,I.ooov,T1old,T1old)
    #L1 += fac*einsum('jlca,kilb,bj,ck->ia',L2old,I.ooov,T1old,T1old)
    # C
    #L1 -= 0.5*fac*einsum('jkad,idbc,bj,ck->ia',L2old,I.ovvv,T1old,T1old)
    L1 += 0.5*fac*einsum('jkad,dibc,bj,ck->ia',L2old,I.vovv,T1old,T1old)
    # D
    #L1 += 0.5*fac*einsum('ilbc,jkal,bj,ck->ia',L2old,I.oovo,T1old,T1old)
    L1 -= 0.5*fac*einsum('ilbc,jkla,bj,ck->ia',L2old,I.ooov,T1old,T1old)

def _LS_LDTSD(L1, I, L2old, T1old, T2old, fac=1.0):
    # A
    L1 -= 0.5*fac*einsum('klca,ijdb,bj,cdkl->ia',L2old,I.oovv,T1old,T2old)
    # B
    L1 -= 0.5*fac*einsum('kicd,ljab,bj,cdkl->ia',L2old,I.oovv,T1old,T2old)
    # C
    L1 -= fac*einsum('jlad,ikbc,bj,cdkl->ia',L2old,I.oovv,T1old,T2old)
    # D
    L1 -= fac*einsum('ilbd,jkac,bj,cdkl->ia',L2old,I.oovv,T1old,T2old)
    # E
    L1 += 0.25*fac*einsum('klab,ijcd,bj,cdkl->ia',L2old,I.oovv,T1old,T2old)
    # F
    L1 += 0.25*fac*einsum('ijcd,klab,bj,cdkl->ia',L2old,I.oovv,T1old,T2old)
    # G
    L1 -= 0.5*fac*einsum('klcb,jida,bj,cdkl->ia',L2old,I.oovv,T1old,T2old)
    # H
    L1 -= 0.5*fac*einsum('kjcd,liba,bj,cdkl->ia',L2old,I.oovv,T1old,T2old)

def _LS_LDTSSS(L1, I, L2old, T1old, fac=1.0):
    # A
    L1 += 0.5*fac*einsum('jlac,ikbd,ck,dl,bj->ia',L2old,I.oovv,T1old,T1old,T1old)
    # B
    L1 += 0.5*fac*einsum('ilbc,jkad,ck,dl,bj->ia',L2old,I.oovv,T1old,T1old,T1old)

def _LD_LS(L2, F, I, L1old, fac=1.0):
    # A
    L2 += fac*einsum('jb,ia->ijab',F.ov,L1old)
    L2 -= fac*einsum('ib,ja->ijab',F.ov,L1old)
    L2 -= fac*einsum('ja,ib->ijab',F.ov,L1old)
    L2 += fac*einsum('ia,jb->ijab',F.ov,L1old)
    # B
    L2 += fac*einsum('cjab,ic->ijab',I.vovv,L1old)
    L2 -= fac*einsum('ciab,jc->ijab',I.vovv,L1old)
    # C
    L2 -= fac*einsum('ijkb,ka->ijab',I.ooov,L1old)
    L2 += fac*einsum('ijka,kb->ijab',I.ooov,L1old)

def _LD_LSTS(L2, F, I, L1old, T1old, fac=1.0):
    # A
    L2 += fac*einsum('jb,ikac,ck->ijab',L1old,I.oovv,T1old)
    L2 -= fac*einsum('ib,jkac,ck->ijab',L1old,I.oovv,T1old)
    L2 -= fac*einsum('ja,ikbc,ck->ijab',L1old,I.oovv,T1old)
    L2 += fac*einsum('ia,jkbc,ck->ijab',L1old,I.oovv,T1old)
    # B
    L2 -= fac*einsum('ic,kjab,ck->ijab',L1old,I.oovv,T1old)
    L2 += fac*einsum('jc,kiab,ck->ijab',L1old,I.oovv,T1old)
    # C
    L2 -= fac*einsum('ka,ijcb,ck->ijab',L1old,I.oovv,T1old)
    L2 += fac*einsum('kb,ijca,ck->ijab',L1old,I.oovv,T1old)

def _LD_LD(L2, F, I, L2old, fac=1.0):
    # A
    L2 += fac*einsum('cb,ijac->ijab',F.vv,L2old)
    L2 -= fac*einsum('ca,ijbc->ijab',F.vv,L2old)
    # B
    L2 -= fac*einsum('jk,ikab->ijab',F.oo,L2old)
    L2 += fac*einsum('ik,jkab->ijab',F.oo,L2old)
    # C
    L2 += 0.5*fac*einsum('cdab,ijcd->ijab',I.vvvv,L2old)
    # D
    L2 += 0.5*fac*einsum('ijkl,klab->ijab',I.oooo,L2old)
    # E
    #L2 += fac*einsum('ikac,cjkb->ijab',L2old,I.voov)
    L2 -= fac*einsum('cjbk,ikac->ijab',I.vovo,L2old)
    L2 += fac*einsum('cibk,jkac->ijab',I.vovo,L2old)
    L2 += fac*einsum('cjak,ikbc->ijab',I.vovo,L2old)
    L2 -= fac*einsum('ciak,jkbc->ijab',I.vovo,L2old)

def _LD_LDTS(L2, F, I, L2old, T1old, fac=1.0):
    # A
    L2 -= fac*einsum('ikab,jc,ck->ijab',L2old,F.ov,T1old)
    L2 += fac*einsum('jkab,ic,ck->ijab',L2old,F.ov,T1old)
    # B
    L2 -= fac*einsum('ijac,kb,ck->ijab',L2old,F.ov,T1old)
    L2 += fac*einsum('ijbc,ka,ck->ijab',L2old,F.ov,T1old)
    # C
    L2 += fac*einsum('ijad,dkbc,ck->ijab',L2old,I.vovv,T1old)
    L2 -= fac*einsum('ijbd,dkac,ck->ijab',L2old,I.vovv,T1old)
    # D
    L2 -= fac*einsum('ilab,jklc,ck->ijab',L2old,I.ooov,T1old)
    L2 += fac*einsum('jlab,iklc,ck->ijab',L2old,I.ooov,T1old)
    # E
    L2 += fac*einsum('ikad,djcb,ck->ijab',L2old,I.vovv,T1old)
    L2 -= fac*einsum('jkad,dicb,ck->ijab',L2old,I.vovv,T1old)
    L2 -= fac*einsum('ikbd,djca,ck->ijab',L2old,I.vovv,T1old)
    L2 += fac*einsum('jkbd,dica,ck->ijab',L2old,I.vovv,T1old)
    # F
    L2 -= fac*einsum('ilac,kjlb,ck->ijab',L2old,I.ooov,T1old)
    L2 += fac*einsum('jlac,kilb,ck->ijab',L2old,I.ooov,T1old)
    L2 += fac*einsum('ilbc,kjla,ck->ijab',L2old,I.ooov,T1old)
    L2 -= fac*einsum('jlbc,kila,ck->ijab',L2old,I.ooov,T1old)
    # G
    #L2 -= fac*einsum('ijcd,kdab,ck->ijab',L2old,I.ovvv,T1old)
    L2 += fac*einsum('ijcd,dkab,ck->ijab',L2old,I.vovv,T1old)
    # H
    #L2 += fac*einsum('klab,ijcl,ck->ijab',L2old,I.oovo,T1old)
    L2 -= fac*einsum('klab,ijlc,ck->ijab',L2old,I.ooov,T1old)

def _LD_LDTD(L2, I, L2old, T2old, fac=1.0):
    # A
    L2 -= 0.5*fac*einsum('jlcd,cdkl,ikab->ijab',I.oovv,T2old,L2old)
    L2 += 0.5*fac*einsum('ilcd,cdkl,jkab->ijab',I.oovv,T2old,L2old)
    # B
    L2 -= 0.5*fac*einsum('klbd,cdkl,ijac->ijab',I.oovv,T2old,L2old)
    L2 += 0.5*fac*einsum('klad,cdkl,ijbc->ijab',I.oovv,T2old,L2old)
    # C
    L2 += fac*einsum('ljdb,cdkl,ikac->ijab',I.oovv,T2old,L2old)
    L2 -= fac*einsum('lidb,cdkl,jkac->ijab',I.oovv,T2old,L2old)
    L2 -= fac*einsum('ljda,cdkl,ikbc->ijab',I.oovv,T2old,L2old)
    L2 += fac*einsum('lida,cdkl,jkbc->ijab',I.oovv,T2old,L2old)
    # D
    L2 -= 0.5*fac*einsum('ijdb,cdkl,klca->ijab',I.oovv,T2old,L2old)
    L2 += 0.5*fac*einsum('ijda,cdkl,klcb->ijab',I.oovv,T2old,L2old)
    # E
    L2 -= 0.5*fac*einsum('ljab,cdkl,kicd->ijab',I.oovv,T2old,L2old)
    L2 += 0.5*fac*einsum('liab,cdkl,kjcd->ijab',I.oovv,T2old,L2old)
    # F
    L2 += 0.25*fac*einsum('ijcd,cdkl,klab->ijab',I.oovv,T2old,L2old)
    # G
    L2 += 0.25*fac*einsum('klab,cdkl,ijcd->ijab',I.oovv,T2old,L2old)

def _LD_LDTSS(L2, F, I, L2old, T1old, fac=1.0):
    # A
    L2 -= fac*einsum('ikab,jlcd,ck,dl->ijab',L2old,I.oovv,T1old,T1old)
    L2 += fac*einsum('jkab,ilcd,ck,dl->ijab',L2old,I.oovv,T1old,T1old)
    # B
    L2 -= fac*einsum('ijac,klbd,ck,dl->ijab',L2old,I.oovv,T1old,T1old)
    L2 += fac*einsum('ijbc,klad,ck,dl->ijab',L2old,I.oovv,T1old,T1old)
    # C
    L2 -= fac*einsum('ikad,ljcb,ck,dl->ijab',L2old,I.oovv,T1old,T1old)
    L2 += fac*einsum('jkad,licb,ck,dl->ijab',L2old,I.oovv,T1old,T1old)
    L2 += fac*einsum('ikbd,ljca,ck,dl->ijab',L2old,I.oovv,T1old,T1old)
    L2 -= fac*einsum('jkbd,lica,ck,dl->ijab',L2old,I.oovv,T1old,T1old)
    # D
    L2 += 0.5*fac*einsum('klab,ijcd,ck,dl->ijab',L2old,I.oovv,T1old,T1old)
    # E
    L2 += 0.5*fac*einsum('ijcd,klab,ck,dl->ijab',L2old,I.oovv,T1old,T1old)

class lambda_int(object):
    def  __init__(self, F, I, T1old, T2old):
        TTemp = 0.5*T2old + einsum('bj,ck->bcjk',T1old,T1old)
        IvovoT1 = einsum('aibc,ck->aibk',I.vovv,T1old)
        IvovoT2 = einsum('bj,jika->biak',T1old,I.ooov)
        self.IooovT = einsum('ikbc,bj->ikjc',I.oovv,T1old)
        self.IvovvT = einsum('ck,kiab->ciab',T1old,I.oovv)
        self.IovT = einsum('ikac,ck->ia',I.oovv,T1old)
        self.IToo = einsum('ib,bj->ij',F.ov,T1old) + F.oo\
            + einsum('ikbc,bcjk->ij',I.oovv,TTemp)\
            + einsum('ijkb,bj->ik',I.ooov,T1old)
        self.ITvv = einsum('ja,bj->ba',F.ov,T1old) - F.vv\
            - einsum('bkac,ck->ba',I.vovv,T1old)\
            + einsum('jkac,bcjk->ba',I.oovv,TTemp)
        self.ITvovo = -IvovoT1 - IvovoT2 - I.vovo
        self.IToovv = einsum('kica,bcjk->ijab',I.oovv,T2old)\
            - einsum('ciba,bj->ijac',self.IvovvT,T1old)
        self.ITvooo = einsum('cibk,bj->cijk',I.vovo,T1old) + 0.5*I.vooo\
            + 0.5*einsum('ic,bcjk->bijk',F.ov,T2old)\
            + 0.5*einsum('ijkl,bj->bikl',I.oooo,T1old)\
            + einsum('kilc,bcjk->bilj',I.ooov,T2old)\
            - einsum('bicl,ck->bilk',IvovoT2,T1old)\
            + 0.5*einsum('dibc,bcjk->dijk',I.vovv,TTemp)\
            + 0.5*einsum('id,cdkl->cikl',self.IovT,T2old)\
            - einsum('ikjc,cdkl->dijl',self.IooovT,T2old)\
            - 0.5*einsum('bicd,cdkl->bikl',self.IvovvT,TTemp)
        self.ITvvvo = einsum('cjak,bj->cbak',I.vovo,T1old) - 0.5*I.vvvo\
            - 0.5*einsum('ka,bcjk->bcaj',F.ov,T2old)\
            - 0.5*einsum('cdab,bj->cdaj',I.vvvv,T1old)\
            - einsum('dkca,bcjk->bdaj',I.vovv,T2old)\
            - einsum('djak,bj->bdak',IvovoT1,T1old)\
            + 0.5*einsum('jkla,bcjk->bcal',I.ooov,TTemp)\
            - 0.5*einsum('la,cdkl->cdak',self.IovT,T2old)\
            + einsum('bkac,cdkl->bdal',self.IvovvT,T2old)\
            + 0.5*einsum('klja,cdkl->cdaj',self.IooovT,TTemp)
        self.ITov = self.IovT + F.ov
        self.ITvovv = self.IvovvT - I.vovv
        self.ITooov = self.IooovT + I.ooov
        self.ITvovo2 = - IvovoT1 - IvovoT2 - I.vovo\
                - einsum('djcb,ck->djbk',self.IvovvT,T1old)
        self.ITvvvv = einsum('dkab,ck->cdab',I.vovv,T1old)\
            + 0.5*einsum('klab,cdkl->cdab',I.oovv,TTemp)\
            + 0.5*I.vvvv
        self.IToooo = einsum('ijlc,ck->ijkl',I.ooov,T1old)\
            - 0.5*einsum('ijcd,cdkl->ijkl',I.oovv,TTemp)\
            - 0.5*I.oooo
        self.IToovv2 = einsum('ljdb,cdkl->kjcb',I.oovv,T2old)

def _Lambda_opt(L1, L2, F, I, L1old, L2old, T1old, T2old, fac=1.0):
    t1 = time.time()

    TTemp = 0.5*T2old + einsum('bj,ck->bcjk',T1old,T1old)
    IvovoT1 = einsum('aibc,ck->aibk',I.vovv,T1old)
    IvovoT2 = einsum('bj,jika->biak',T1old,I.ooov)
    IooovT = einsum('ikbc,bj->ikjc',I.oovv,T1old)
    IvovvT = einsum('ck,kiab->ciab',T1old,I.oovv)
    IovT = einsum('ikac,ck->ia',I.oovv,T1old)

    ## OO
    IToo = einsum('ib,bj->ij',F.ov,T1old) + F.oo\
        + einsum('ikbc,bcjk->ij',I.oovv,TTemp)\
        + einsum('ijkb,bj->ik',I.ooov,T1old)
    L1 -= fac*einsum('ja,ij->ia',L1old,IToo)

    temp = fac*einsum('ikab,jk->ijab',L2old,IToo)
    L2 -= temp
    L2 += temp.transpose((1,0,2,3))
    IToo = None
    temp = None

    ## VV
    ITvv = einsum('ja,bj->ba',F.ov,T1old) - F.vv\
        - einsum('bkac,ck->ba',I.vovv,T1old)\
        + einsum('jkac,bcjk->ba',I.oovv,TTemp)
    L1 -= fac*einsum('ib,ba->ia',L1old,ITvv)

    temp = fac*einsum('ijac,cb->ijab',L2old,ITvv)
    L2 -= temp
    L2 += temp.transpose((0,1,3,2))
    ITvv = None
    temp = None

    ## VOVO
    ITvovo = -IvovoT1 - IvovoT2 - I.vovo
    L1 += fac*einsum('jb,biaj->ia',L1old,ITvovo)
    ITvovo = None

    ## OOVV
    IToovv = einsum('kica,bcjk->ijab',I.oovv,T2old)\
        - einsum('ciba,bj->ijac',IvovvT,T1old)
    L1 += fac*einsum('jb,ijab->ia',L1old,IToovv)
    IToovv = None

    ## VOOO
    ITvooo = einsum('cibk,bj->cijk',I.vovo,T1old) + 0.5*I.vooo\
        + 0.5*einsum('ic,bcjk->bijk',F.ov,T2old)\
        + 0.5*einsum('ijkl,bj->bikl',I.oooo,T1old)\
        + einsum('kilc,bcjk->bilj',I.ooov,T2old)\
        - einsum('bicl,ck->bilk',IvovoT2,T1old)\
        + 0.5*einsum('dibc,bcjk->dijk',I.vovv,TTemp)\
        + 0.5*einsum('id,cdkl->cikl',IovT,T2old)\
        - einsum('ikjc,cdkl->dijl',IooovT,T2old)\
        - 0.5*einsum('bicd,cdkl->bikl',IvovvT,TTemp)
    L1 += fac*einsum('jkab,bijk->ia',L2old,ITvooo)
    ITvooo = None

    # VVVO
    ITvvvo = einsum('cjak,bj->cbak',I.vovo,T1old) - 0.5*I.vvvo\
        - 0.5*einsum('ka,bcjk->bcaj',F.ov,T2old)\
        - 0.5*einsum('cdab,bj->cdaj',I.vvvv,T1old)\
        - einsum('dkca,bcjk->bdaj',I.vovv,T2old)\
        - einsum('djak,bj->bdak',IvovoT1,T1old)\
        + 0.5*einsum('jkla,bcjk->bcal',I.ooov,TTemp)\
        - 0.5*einsum('la,cdkl->cdak',IovT,T2old)\
        + einsum('bkac,cdkl->bdal',IvovvT,T2old)\
        + 0.5*einsum('klja,cdkl->cdaj',IooovT,TTemp)
    L1 += fac*einsum('ikbc,cbak->ia',L2old,ITvvvo)
    ITvvvo = None

    ## OV
    ITov = IovT + F.ov
    temp = fac*einsum('jb,ia->ijab',L1old,ITov)
    L2 += temp
    L2 -= temp.transpose((1,0,2,3))
    L2 -= temp.transpose((0,1,3,2))
    L2 += temp.transpose((1,0,3,2))
    ITov = None

    ## VOVV
    ITvovv = IvovvT - I.vovv
    temp = fac*einsum('ic,cjab->ijab',L1old,ITvovv)
    L2 -= temp
    L2 += temp.transpose((1,0,2,3))
    ITvovv = None

    ## OOOV
    ITooov = IooovT + I.ooov
    temp = fac*einsum('ka,ijkb->ijab',L1old,ITooov)
    L2 -= temp
    L2 += temp.transpose((0,1,3,2))
    ITooov = None

    ## VOVO
    ITvovo = - IvovoT1 - IvovoT2 - I.vovo\
            - einsum('djcb,ck->djbk',IvovvT,T1old)
    temp = fac*einsum('ikad,djbk->ijab',L2old,ITvovo)
    L2 += temp
    L2 -= temp.transpose((1,0,2,3))
    L2 -= temp.transpose((0,1,3,2))
    L2 += temp.transpose((1,0,3,2))
    ITvovo = None

    ## VVVV
    ITvvvv = einsum('dkab,ck->cdab',I.vovv,T1old)\
        + 0.5*einsum('klab,cdkl->cdab',I.oovv,TTemp)\
        + 0.5*I.vvvv
    L2 += fac*einsum('ijcd,cdab->ijab',L2old,ITvvvv)
    ITvvvv = None

    ## OOOO
    IToooo = einsum('ijlc,ck->ijkl',I.ooov,T1old)\
        - 0.5*einsum('ijcd,cdkl->ijkl',I.oovv,TTemp)\
        - 0.5*I.oooo
    L2 -= fac*einsum('klab,ijkl->ijab',L2old,IToooo)
    IToooo = None

    ## OOVV
    IToovv = einsum('ljdb,cdkl->kjcb',I.oovv,T2old)
    temp = fac*einsum('ikac,kjcb->ijab',L2old,IToovv)
    L2 += temp
    L2 -= temp.transpose((1,0,2,3))
    L2 -= temp.transpose((0,1,3,2))
    L2 += temp.transpose((1,0,3,2))
    IToovv = None

    ## LT terms
    Ltemp1 = einsum('jkbd,bcjk->cd',L2old,T2old)
    Ltemp2 = einsum('jlbc,bcjk->lk',L2old,T2old)

    L1 += 0.5*fac*einsum('cd,dica->ia',Ltemp1,I.vovv)

    L1 -= 0.5*fac*einsum('lk,kila->ia',Ltemp2,I.ooov)

    L1 -= 0.5*fac*einsum('db,bida->ia',Ltemp1,IvovvT)

    L1 -= 0.5*fac*einsum('jl,lija->ia',Ltemp2,IooovT)

    temp = 0.5*fac*einsum('da,ijdb->ijab',Ltemp1,I.oovv)
    L2 -= temp
    L2 += temp.transpose((0,1,3,2))

    temp = 0.5*fac*einsum('il,ljab->ijab',Ltemp2,I.oovv)
    L2 -= temp
    L2 += temp.transpose((1,0,2,3))

    t2 = time.time()

def _Lambda_opt_int(L1, L2, F, I, L1old, L2old, T1old, T2old, intor, fac=1.0):
    ## OO
    L1 -= fac*einsum('ja,ij->ia',L1old,intor.IToo)

    temp = fac*einsum('ikab,jk->ijab',L2old,intor.IToo)
    L2 -= temp
    L2 += temp.transpose((1,0,2,3))

    ## VV
    L1 -= fac*einsum('ib,ba->ia',L1old,intor.ITvv)

    temp = fac*einsum('ijac,cb->ijab',L2old,intor.ITvv)
    L2 -= temp
    L2 += temp.transpose((0,1,3,2))

    ## VOVO
    L1 += fac*einsum('jb,biaj->ia',L1old,intor.ITvovo)

    ## OOVV
    L1 += fac*einsum('jb,ijab->ia',L1old,intor.IToovv)

    ## VOOO
    L1 += fac*einsum('jkab,bijk->ia',L2old,intor.ITvooo)

    # VVVO
    L1 += fac*einsum('ikbc,cbak->ia',L2old,intor.ITvvvo)

    ## OV
    temp = fac*einsum('jb,ia->ijab',L1old,intor.ITov)
    L2 += temp
    L2 -= temp.transpose((1,0,2,3))
    L2 -= temp.transpose((0,1,3,2))
    L2 += temp.transpose((1,0,3,2))

    ## VOVV
    temp = fac*einsum('ic,cjab->ijab',L1old,intor.ITvovv)
    L2 -= temp
    L2 += temp.transpose((1,0,2,3))

    ## OOOV
    temp = fac*einsum('ka,ijkb->ijab',L1old,intor.ITooov)
    L2 -= temp
    L2 += temp.transpose((0,1,3,2))

    ## VOVO
    temp = fac*einsum('ikad,djbk->ijab',L2old,intor.ITvovo2)
    L2 += temp
    L2 -= temp.transpose((1,0,2,3))
    L2 -= temp.transpose((0,1,3,2))
    L2 += temp.transpose((1,0,3,2))

    ## VVVV
    L2 += fac*einsum('ijcd,cdab->ijab',L2old,intor.ITvvvv)

    ## OOOO
    L2 -= fac*einsum('klab,ijkl->ijab',L2old,intor.IToooo)

    ## OOVV
    temp = fac*einsum('ikac,kjcb->ijab',L2old,intor.IToovv2)
    L2 += temp
    L2 -= temp.transpose((1,0,2,3))
    L2 -= temp.transpose((0,1,3,2))
    L2 += temp.transpose((1,0,3,2))

    ## LT terms
    Ltemp1 = einsum('jkbd,bcjk->cd',L2old,T2old)
    Ltemp2 = einsum('jlbc,bcjk->lk',L2old,T2old)

    L1 += 0.5*fac*einsum('cd,dica->ia',Ltemp1,I.vovv)

    L1 -= 0.5*fac*einsum('lk,kila->ia',Ltemp2,I.ooov)

    L1 -= 0.5*fac*einsum('db,bida->ia',Ltemp1,intor.IvovvT)

    L1 -= 0.5*fac*einsum('jl,lija->ia',Ltemp2,intor.IooovT)

    temp = 0.5*fac*einsum('da,ijdb->ijab',Ltemp1,I.oovv)
    L2 -= temp
    L2 += temp.transpose((0,1,3,2))

    temp = 0.5*fac*einsum('il,ljab->ijab',Ltemp2,I.oovv)
    L2 -= temp
    L2 += temp.transpose((1,0,2,3))

def _uccsd_Lambda_opt(L1a, L1b, L2aa, L2ab, L2bb, Fa, Fb, Ia, Ib, Iabab,
        L1olds, L2olds, T1olds, T2olds, fac=1.0):

    # unpack
    L1aold,L1bold = L1olds
    L2aaold,L2abold,L2bbold = L2olds
    T1aold,T1bold = T1olds
    T2aaold,T2abold,T2bbold = T2olds

    # dims
    noa,nva = L1aold.shape
    nob,nvb = L1bold.shape
    no = noa + nob
    nv = nva + nvb

    t1 = time.time()
    #TTemp = 0.5*T2old + einsum('bj,ck->bcjk',T1old,T1old)
    TTempaa = 0.5*T2aaold + einsum('bj,ck->bcjk',T1aold,T1aold)
    TTempab = 0.5*T2abold + einsum('bj,ck->bcjk',T1aold,T1bold)
    TTempbb = 0.5*T2bbold + einsum('bj,ck->bcjk',T1bold,T1bold)

    #IvovoT1 = einsum('aibc,ck->aibk',I.vovv,T1old)
    Ivovos1 = einsum('aibc,ck->aibk',Ia.vovv,T1aold)
    IVOVOs1 = einsum('aibc,ck->aibk',Ib.vovv,T1bold)
    IvOvOs1 = einsum('aibc,ck->aibk',Iabab.vovv,T1bold)
    IVoVos1 = einsum('iacb,ck->aibk',Iabab.ovvv,T1aold)
    IvOVos1 = -einsum('aicb,ck->aibk',Iabab.vovv,T1aold)
    IVovOs1 = -einsum('iabc,ck->aibk',Iabab.ovvv,T1bold)

    #IvovoT2 = einsum('bj,jika->biak',T1old,I.ooov)
    Ivovos2 = einsum('bj,jika->biak',T1aold,Ia.ooov)
    IVOVOs2 = einsum('bj,jika->biak',T1bold,Ib.ooov)
    IvOvOs2 = -einsum('bj,jiak->biak',T1aold,Iabab.oovo)
    IVoVos2 = -einsum('bj,ijka->biak',T1bold,Iabab.ooov)
    IvOVos2 = einsum('bj,jika->biak',T1aold,Iabab.ooov)
    IVovOs2 = einsum('bj,ijak->biak',T1bold,Iabab.oovo)

    #IvovvT = einsum('ck,kiab->ciab',T1old,I.oovv)
    Ivovvs = einsum('ck,kiab->ciab',T1aold,Ia.oovv)
    IVOVVs = einsum('ck,kiab->ciab',T1bold,Ib.oovv)
    IVoVvs = einsum('ck,ikba->ciab',T1bold,Iabab.oovv)
    IvOvVs = einsum('ck,kiab->ciab',T1aold,Iabab.oovv)
    IvOVvs = -einsum('ck,kiba->ciab',T1aold,Iabab.oovv)
    IVovVs = -einsum('ck,ikab->ciab',T1bold,Iabab.oovv)

    #IooovT = einsum('ikbc,bj->ikjc',I.oovv,T1old)
    Iooovs = einsum('ikbc,bj->ikjc',Ia.oovv,T1aold)
    IOOOVs = einsum('ikbc,bj->ikjc',Ib.oovv,T1bold)
    IOoOvs = einsum('kicb,bj->ikjc',Iabab.oovv,T1bold)
    IoOoVs = einsum('ikbc,bj->ikjc',Iabab.oovv,T1aold)
    IoOOvs = -einsum('ikcb,bj->ikjc',Iabab.oovv,T1bold)
    IOooVs = -einsum('kibc,bj->ikjc',Iabab.oovv,T1aold)

    #IovT = einsum('ikac,ck->ia',I.oovv,T1old)
    Iovs = einsum('ikac,ck->ia',Ia.oovv,T1aold)
    Iovs += einsum('ikac,ck->ia',Iabab.oovv,T1bold)
    IOVs = einsum('ikac,ck->ia',Ib.oovv,T1bold)
    IOVs += einsum('kica,ck->ia',Iabab.oovv,T1aold)

    ## OO
    ITooa = einsum('ib,bj->ij',Fa.ov,T1aold) + Fa.oo\
        + einsum('ikbc,bcjk->ij',Ia.oovv,TTempaa)\
        + einsum('ikbc,bcjk->ij',Iabab.oovv,TTempab)\
        + 0.5*einsum('ikcb,cbjk->ij',Iabab.oovv,T2abold)\
        + einsum('ijkb,bj->ik',Ia.ooov,T1aold)\
        + einsum('ijkb,bj->ik',Iabab.ooov,T1bold)
    L1a -= fac*einsum('ja,ij->ia',L1aold,ITooa)
    tempa = einsum('ikab,jk->ijab',L2aaold,ITooa)
    tempa -= tempa.transpose((1,0,2,3))
    L2aa -= fac*tempa
    tempa = None
    tempab = einsum('kjab,ik->ijab',L2abold,ITooa)
    ITooa = None
    IToob = einsum('ib,bj->ij',Fb.ov,T1bold) + Fb.oo\
        + einsum('ikbc,bcjk->ij',Ib.oovv,TTempbb)\
        + einsum('kicb,cbkj->ij',Iabab.oovv,TTempab)\
        + 0.5*einsum('kibc,bckj->ij',Iabab.oovv,T2abold)\
        + einsum('ijkb,bj->ik',Ib.ooov,T1bold)\
        + einsum('jibk,bj->ik',Iabab.oovo,T1aold)
    L1b -= fac*einsum('ja,ij->ia',L1bold,IToob)
    tempb = einsum('ikab,jk->ijab',L2bbold,IToob)
    tempb -= tempb.transpose((1,0,2,3))
    L2bb -= fac*tempb
    tempb = None
    tempab += einsum('ikab,jk->ijab',L2abold,IToob)
    IToob = None
    L2ab -= fac*tempab
    tempab = None

    ## VV
    ITvva = einsum('ja,bj->ba',Fa.ov,T1aold) - Fa.vv\
        - einsum('bkac,ck->ba',Ia.vovv,T1aold)\
        - einsum('bkac,ck->ba',Iabab.vovv,T1bold)
    ITvva += einsum('jkac,bcjk->ba',Ia.oovv,TTempaa)
    ITvva += einsum('jkac,bcjk->ba',Iabab.oovv,TTempab)
    ITvva += 0.5*einsum('kjac,bckj->ba',Iabab.oovv,T2abold)
    L1a -= fac*einsum('ib,ba->ia',L1aold,ITvva)
    tempa = einsum('ijac,cb->ijab',L2aaold,ITvva)
    tempa -= tempa.transpose((0,1,3,2))
    tempab = einsum('ijcb,ca->ijab',L2abold,ITvva)
    ITvva = None
    ITvvb = einsum('ja,bj->ba',Fb.ov,T1bold) - Fb.vv\
        - einsum('bkac,ck->ba',Ib.vovv,T1bold)\
        - einsum('kbca,ck->ba',Iabab.ovvv,T1aold)
    ITvvb += einsum('jkac,bcjk->ba',Ib.oovv,TTempbb)
    ITvvb += einsum('kjca,cbkj->ba',Iabab.oovv,TTempab)
    ITvvb += 0.5*einsum('jkca,cbjk->ba',Iabab.oovv,T2abold)
    L1b -= fac*einsum('ib,ba->ia',L1bold,ITvvb)
    tempb = einsum('ijac,cb->ijab',L2bbold,ITvvb)
    tempb -= tempb.transpose((0,1,3,2))
    tempab += einsum('ijac,cb->ijab',L2abold,ITvvb)
    ITvvb = None

    L2aa -= fac*tempa
    L2ab -= fac*tempab
    L2bb -= fac*tempb

    ## OOVV
    IToovvs1 = einsum('kida,bdjk->ijab',Ia.oovv,T2aaold)
    IToovvs1 += einsum('ikad,bdjk->ijab',Iabab.oovv,T2abold)
    IToovvs1 -= einsum('ciba,bj->ijac',Ivovvs,T1aold)
    L1a += fac*einsum('jb,ijab->ia',L1aold,IToovvs1)
    IToovvs1 = None

    ITOOVVs1 = einsum('kica,bcjk->ijab',Ib.oovv,T2bbold)
    ITOOVVs1 += einsum('kica,cbkj->ijab',Iabab.oovv,T2abold)
    ITOOVVs1 -= einsum('ciba,bj->ijac',IVOVVs,T1bold)
    L1b += fac*einsum('jb,ijab->ia',L1bold,ITOOVVs1)
    ITOOVVs1 = None

    IToOvVs1 = einsum('kica,cbkj->ijab',Ia.oovv,T2abold)
    IToOvVs1 += einsum('ikac,bcjk->ijab',Iabab.oovv,T2bbold)
    IToOvVs1 -= einsum('ciba,bj->ijac',IVoVvs,T1bold)
    L1a += fac*einsum('jb,ijab->ia',L1bold,IToOvVs1)
    IToOvVs1 = None

    ITOoVvs1 = einsum('kica,bcjk->ijab',Iabab.oovv,T2aaold)
    ITOoVvs1 += einsum('kica,bcjk->ijab',Ib.oovv,T2abold)
    ITOoVvs1 -= einsum('ciba,bj->ijac',IvOvVs,T1aold)
    L1b += fac*einsum('jb,ijab->ia',L1aold,ITOoVvs1)
    ITOoVvs1 = None

    ## VOOO
    ITvooos = einsum('cibk,bj->cijk',Ia.vovo,T1aold)
    ITvooos += 0.5*Ia.vooo
    ITvooos += 0.5*einsum('ic,bcjk->bijk',Fa.ov,T2aaold)
    ITvooos += 0.5*einsum('ijkl,bj->bikl',Ia.oooo,T1aold)
    ITvooos += einsum('kilc,bcjk->bilj',Ia.ooov,T2aaold)
    ITvooos -= einsum('iklc,bcjk->bilj',Iabab.ooov,T2abold)
    ITvooos -= einsum('bicl,ck->bilk',Ivovos2,T1aold)
    ITvooos += 0.5*einsum('dibc,bcjk->dijk',Ia.vovv,TTempaa)
    ITvooos += 0.5*einsum('id,cdkl->cikl',Iovs,T2aaold)
    ITvooos -= einsum('ikjc,cdkl->dijl',Iooovs,T2aaold)
    ITvooos -= einsum('ikjc,dclk->dijl',IoOoVs,T2abold)
    ITvooos -= 0.5*einsum('bicd,cdkl->bikl',Ivovvs,TTempaa)
    L1a += fac*einsum('jkab,bijk->ia',L2aaold,ITvooos)
    ITvooos = None

    ITVOOOs = einsum('cibk,bj->cijk',Ib.vovo,T1bold)
    ITVOOOs += 0.5*Ib.vooo
    ITVOOOs += 0.5*einsum('ic,bcjk->bijk',Fb.ov,T2bbold)
    ITVOOOs += 0.5*einsum('ijkl,bj->bikl',Ib.oooo,T1bold)
    ITVOOOs += einsum('kilc,bcjk->bilj',Ib.ooov,T2bbold)
    ITVOOOs -= einsum('kicl,cbkj->bilj',Iabab.oovo,T2abold)
    ITVOOOs -= einsum('bicl,ck->bilk',IVOVOs2,T1bold)
    ITVOOOs += 0.5*einsum('dibc,bcjk->dijk',Ib.vovv,TTempbb)
    ITVOOOs += 0.5*einsum('id,cdkl->cikl',IOVs,T2bbold)
    ITVOOOs -= einsum('ikjc,cdkl->dijl',IOOOVs,T2bbold)
    ITVOOOs -= einsum('ikjc,cdkl->dijl',IOoOvs,T2abold)
    ITVOOOs -= 0.5*einsum('bicd,cdkl->bikl',IVOVVs,TTempbb)
    L1b += fac*einsum('jkab,bijk->ia',L2bbold,ITVOOOs)
    ITVOOOs = None

    ITVooOs = -einsum('icbk,bj->cijk',Iabab.ovvo,T1aold)
    ITVooOs -= 0.5*Iabab.ovoo.transpose((1,0,2,3))
    ITVooOs -= 0.5*einsum('ic,cbjk->bijk',Fa.ov,T2abold)
    ITVooOs += 0.5*einsum('ijkl,bj->bikl',Iabab.oooo,T1bold)
    ITVooOs += einsum('kilc,cbkj->bilj',Ia.ooov,T2abold)
    ITVooOs -= einsum('iklc,bcjk->bilj',Iabab.ooov,T2bbold)
    ITVooOs -= einsum('bicl,ck->bilk',IVoVos2,T1bold)
    ITVooOs -= 0.5*einsum('idbc,bcjk->dijk',Iabab.ovvv,TTempab)
    ITVooOs -= 0.25*einsum('idcb,cbjk->dijk',Iabab.ovvv,T2abold)
    ITVooOs -= 0.5*einsum('id,dckl->cikl',Iovs,T2abold)
    ITVooOs -= einsum('ikjc,cdkl->dijl',Iooovs,T2abold)
    ITVooOs -= einsum('ikjc,cdkl->dijl',IoOoVs,T2bbold)
    ITVooOs -= 0.5*einsum('bicd,cdkl->bikl',IVovVs,TTempab)
    ITVooOs += 0.25*einsum('bicd,dckl->bikl',IVoVvs,T2abold)
    L1a += fac*einsum('jkab,bijk->ia',L2abold,ITVooOs)
    ITVooOs = None

    ITvOOos = -einsum('cikb,bj->cijk',Iabab.voov,T1bold)
    ITvOOos -= 0.5*Iabab.vooo.transpose((0,1,3,2))
    ITvOOos -= 0.5*einsum('ic,bckj->bijk',Fb.ov,T2abold)
    ITvOOos += 0.5*einsum('jilk,bj->bikl',Iabab.oooo,T1aold)
    ITvOOos += einsum('kilc,bcjk->bilj',Ib.ooov,T2abold)
    ITvOOos -= einsum('kicl,bcjk->bilj',Iabab.oovo,T2aaold)
    ITvOOos -= einsum('bicl,ck->bilk',IvOvOs2,T1aold)
    ITvOOos -= 0.5*einsum('dicb,cbkj->dijk',Iabab.vovv,TTempab)
    ITvOOos -= 0.25*einsum('dibc,bckj->dijk',Iabab.vovv,T2abold)
    ITvOOos -= 0.5*einsum('id,cdlk->cikl',IOVs,T2abold)
    ITvOOos -= einsum('ikjc,cdkl->dijl',IOoOvs,T2aaold)
    ITvOOos -= einsum('ikjc,dclk->dijl',IOOOVs,T2abold)
    ITvOOos -= 0.5*einsum('bicd,dclk->bikl',IvOVvs,TTempab)
    ITvOOos += 0.25*einsum('bicd,cdlk->bikl',IvOvVs,T2abold)
    L1b += fac*einsum('kjba,bijk->ia',L2abold,ITvOOos)
    ITvOOos = None

    ITVoOos = einsum('ickb,bj->cijk',Iabab.ovov,T1bold)
    ITVoOos += 0.5*Iabab.ovoo.transpose((1,0,3,2))
    ITVoOos += 0.5*einsum('ic,cbkj->bijk',Fa.ov,T2abold)
    ITVoOos -= 0.5*einsum('ijlk,bj->bikl',Iabab.oooo,T1bold)
    ITVoOos -= einsum('ikcl,cbjk->bilj',Iabab.oovo,T2abold)
    ITVoOos -= einsum('bicl,ck->bilk',IVovOs2,T1aold)
    ITVoOos += 0.5*einsum('idcb,cbkj->dijk',Iabab.ovvv,TTempab)
    ITVoOos += 0.25*einsum('idbc,bckj->dijk',Iabab.ovvv,T2abold)
    ITVoOos += 0.5*einsum('id,dclk->cikl',Iovs,T2abold)
    ITVoOos += einsum('ikjc,cdlk->dijl',IoOOvs,T2abold)
    ITVoOos -= 0.5*einsum('bicd,dclk->bikl',IVoVvs,TTempab)
    ITVoOos += 0.25*einsum('bicd,cdlk->bikl',IVovVs,T2abold)
    L1a -= fac*einsum('kjab,bijk->ia',L2abold,ITVoOos)
    ITVoOos = None

    ITvOoOs = einsum('cibk,bj->cijk',Iabab.vovo,T1aold)
    ITvOoOs += 0.5*Iabab.vooo
    ITvOoOs += 0.5*einsum('ic,bcjk->bijk',Fb.ov,T2abold)
    ITvOoOs -= 0.5*einsum('jikl,bj->bikl',Iabab.oooo,T1aold)
    ITvOoOs -= einsum('kilc,bckj->bilj',Iabab.ooov,T2abold)
    ITvOoOs -= einsum('bicl,ck->bilk',IvOVos2,T1bold)
    ITvOoOs += 0.5*einsum('dibc,bcjk->dijk',Iabab.vovv,TTempab)
    ITvOoOs += 0.25*einsum('dicb,cbjk->dijk',Iabab.vovv,T2abold)
    ITvOoOs += 0.5*einsum('id,cdkl->cikl',IOVs,T2abold)
    ITvOoOs += einsum('ikjc,dckl->dijl',IOooVs,T2abold)
    ITvOoOs -= 0.5*einsum('bicd,cdkl->bikl',IvOvVs,TTempab)
    ITvOoOs += 0.25*einsum('bicd,dckl->bikl',IvOVvs,T2abold)
    L1b -= fac*einsum('jkba,bijk->ia',L2abold,ITvOoOs)
    ITvOoOs = None

    # VVVO
    ITvvvos =  - 0.5*Ia.vvvo
    ITvvvos += einsum('cjak,bj->cbak',Ia.vovo,T1aold)
    ITvvvos -= 0.5*einsum('ka,bcjk->bcaj',Fa.ov,T2aaold)
    ITvvvos -= 0.5*einsum('cdab,bj->cdaj',Ia.vvvv,T1aold)
    ITvvvos -= einsum('dkca,bcjk->bdaj',Ia.vovv,T2aaold)
    ITvvvos += einsum('dkac,bcjk->bdaj',Iabab.vovv,T2abold)
    ITvvvos -= einsum('djak,bj->bdak',Ivovos1,T1aold)
    ITvvvos += 0.5*einsum('jkla,bcjk->bcal',Ia.ooov,TTempaa)
    ITvvvos -= 0.5*einsum('la,cdkl->cdak',Iovs,T2aaold)
    ITvvvos += einsum('bkac,cdkl->bdal',Ivovvs,T2aaold)
    ITvvvos += einsum('bkac,dclk->bdal',IvOvVs,T2abold)
    ITvvvos += 0.5*einsum('klja,cdkl->cdaj',Iooovs,TTempaa)
    L1a += fac*einsum('ikbc,cbak->ia',L2aaold,ITvvvos)
    ITvvvos = None

    ITVVVOs =  - 0.5*Ib.vvvo
    ITVVVOs += einsum('cjak,bj->cbak',Ib.vovo,T1bold)
    ITVVVOs -= 0.5*einsum('ka,bcjk->bcaj',Fb.ov,T2bbold)
    ITVVVOs -= 0.5*einsum('cdab,bj->cdaj',Ib.vvvv,T1bold)
    ITVVVOs -= einsum('dkca,bcjk->bdaj',Ib.vovv,T2bbold)
    ITVVVOs += einsum('kdca,cbkj->bdaj',Iabab.ovvv,T2abold)
    ITVVVOs -= einsum('djak,bj->bdak',IVOVOs1,T1bold)
    ITVVVOs += 0.5*einsum('jkla,bcjk->bcal',Ib.ooov,TTempbb)
    ITVVVOs -= 0.5*einsum('la,cdkl->cdak',IOVs,T2bbold)
    ITVVVOs += einsum('bkac,cdkl->bdal',IVOVVs,T2bbold)
    ITVVVOs += einsum('bkac,cdkl->bdal',IVoVvs,T2abold)
    ITVVVOs += 0.5*einsum('klja,cdkl->cdaj',IOOOVs,TTempbb)
    L1b += fac*einsum('ikbc,cbak->ia',L2bbold,ITVVVOs)
    ITVVVOs = None

    ITVvvOs =  + 0.5*Iabab.vvvo.transpose((1,0,2,3))
    ITVvvOs -= einsum('jcak,bj->cbak',Iabab.ovvo,T1aold)
    ITVvvOs -= 0.5*einsum('ka,cbkj->bcaj',Fa.ov,T2abold)
    ITVvvOs += 0.5*einsum('dcab,bj->cdaj',Iabab.vvvv,T1bold)
    ITVvvOs -= einsum('dkca,cbkj->bdaj',Ia.vovv,T2abold)
    ITVvvOs += einsum('dkac,bcjk->bdaj',Iabab.vovv,T2bbold)
    ITVvvOs -= einsum('djak,bj->bdak',IvOvOs1,T1bold)
    ITVvvOs += 0.5*einsum('kjal,cbkj->bcal',Iabab.oovo,TTempab)
    ITVvvOs += 0.25*einsum('jkal,cbjk->bcal',Iabab.oovo,T2abold)
    ITVvvOs -= 0.5*einsum('la,dclk->cdak',Iovs,T2abold)
    ITVvvOs -= einsum('bkac,dckl->bdal',IVovVs,T2abold)
    ITVvvOs += 0.5*einsum('klja,dclk->cdaj',IOoOvs,TTempab)
    ITVvvOs -= 0.25*einsum('klja,dckl->cdaj',IoOOvs,T2abold)
    L1a += fac*einsum('ikbc,cbak->ia',L2abold,ITVvvOs)
    ITVvvOs = None

    ITvVVos =  + 0.5*Iabab.vvov.transpose((0,1,3,2))
    ITvVVos -= einsum('cjka,bj->cbak',Iabab.voov,T1bold)
    ITvVVos -= 0.5*einsum('ka,bcjk->bcaj',Fb.ov,T2abold)
    ITvVVos += 0.5*einsum('cdba,bj->cdaj',Iabab.vvvv,T1aold)
    ITvVVos += einsum('kdca,bcjk->bdaj',Iabab.ovvv,T2aaold)
    ITvVVos -= einsum('dkca,bcjk->bdaj',Ib.vovv,T2abold)
    ITvVVos -= einsum('djak,bj->bdak',IVoVos1,T1aold)
    ITvVVos += 0.5*einsum('jkla,bcjk->bcal',Iabab.ooov,TTempab)
    ITvVVos += 0.25*einsum('kjla,bckj->bcal',Iabab.ooov,T2abold)
    ITvVVos -= 0.5*einsum('la,cdkl->cdak',IOVs,T2abold)
    ITvVVos -= einsum('bkac,cdlk->bdal',IvOVvs,T2abold)
    ITvVVos += 0.5*einsum('klja,cdkl->cdaj',IoOoVs,TTempab)
    ITvVVos -= 0.25*einsum('klja,cdlk->cdaj',IOooVs,T2abold)
    L1b += fac*einsum('kicb,cbak->ia',L2abold,ITvVVos)
    ITvVVos = None

    ITVvVos =  - 0.5*Iabab.vvov.transpose((1,0,3,2))
    ITVvVos += einsum('jcka,bj->cbak',Iabab.ovov,T1aold)
    ITVvVos += 0.5*einsum('ka,cbjk->bcaj',Fb.ov,T2abold)
    ITVvVos -= 0.5*einsum('dcba,bj->cdaj',Iabab.vvvv,T1aold)
    ITVvVos += einsum('dkca,cbjk->bdaj',Iabab.vovv,T2abold)
    ITVvVos -= einsum('djak,bj->bdak',IvOVos1,T1bold)
    ITVvVos -= 0.5*einsum('kjla,cbkj->bcal',Iabab.ooov,TTempab)
    ITVvVos -= 0.25*einsum('jkla,cbjk->bcal',Iabab.ooov,T2abold)
    ITVvVos += 0.5*einsum('la,dckl->cdak',IOVs,T2abold)
    ITVvVos += einsum('bkac,cdkl->bdal',IVoVvs,T2aaold)
    ITVvVos += einsum('bkac,dclk->bdal',IVOVVs,T2abold)
    ITVvVos += 0.5*einsum('klja,dclk->cdaj',IOooVs,TTempab)
    ITVvVos -= 0.25*einsum('klja,dckl->cdaj',IoOoVs,T2abold)
    L1b -= fac*einsum('kibc,cbak->ia',L2abold,ITVvVos)
    ITVvVos = None

    ITvVvOs = - 0.5*Iabab.vvvo
    ITvVvOs += einsum('cjak,bj->cbak',Iabab.vovo,T1bold)
    ITvVvOs += 0.5*einsum('ka,bckj->bcaj',Fa.ov,T2abold)
    ITvVvOs -= 0.5*einsum('cdab,bj->cdaj',Iabab.vvvv,T1bold)
    ITvVvOs += einsum('kdac,bckj->bdaj',Iabab.ovvv,T2abold)
    ITvVvOs -= einsum('djak,bj->bdak',IVovOs1,T1aold)
    ITvVvOs -= 0.5*einsum('jkal,bcjk->bcal',Iabab.oovo,TTempab)
    ITvVvOs -= 0.25*einsum('kjal,bckj->bcal',Iabab.oovo,T2abold)
    ITvVvOs += 0.5*einsum('la,cdlk->cdak',Iovs,T2abold)
    ITvVvOs += einsum('bkac,cdkl->bdal',Ivovvs,T2abold)
    ITvVvOs += einsum('bkac,cdkl->bdal',IvOvVs,T2bbold)
    ITvVvOs += 0.5*einsum('klja,cdkl->cdaj',IoOOvs,TTempab)
    ITvVvOs -= 0.25*einsum('klja,cdlk->cdaj',IOoOvs,T2abold)
    L1a -= fac*einsum('ikcb,cbak->ia',L2abold,ITvVvOs)
    ITvVvOs = None

    ## OV
    ITovs = Iovs + Fa.ov
    tempaa = einsum('jb,ia->ijab',L1aold,ITovs)
    tempab = einsum('jb,ia->ijab',L1bold,ITovs)
    ITovs = None
    ITOVs = IOVs + Fb.ov
    tempaa += tempaa.transpose((1,0,3,2)) - tempaa.transpose((0,1,3,2)) - tempaa.transpose((1,0,2,3))
    tempbb = einsum('jb,ia->ijab',L1bold,ITOVs)
    tempbb += tempbb.transpose((1,0,3,2)) - tempbb.transpose((0,1,3,2)) - tempbb.transpose((1,0,2,3))
    tempab += einsum('ia,jb->ijab',L1aold,ITOVs)
    ITOVs = None
    L2aa += fac*tempaa
    L2ab += fac*tempab
    L2bb += fac*tempbb

    ## VOVV
    ITvovvs = Ivovvs - Ia.vovv
    tempaa = einsum('ic,cjab->ijab',L1aold,ITvovvs)
    ITvovvs = None
    tempaa -= tempaa.transpose((1,0,2,3))
    L2aa -= fac*tempaa

    ITVOVVs = IVOVVs - Ib.vovv
    tempbb = einsum('ic,cjab->ijab',L1bold,ITVOVVs)
    ITVOVVs = None 
    tempbb -= tempbb.transpose((1,0,2,3))
    L2bb -= fac*tempbb

    ITvOvVs = IvOvVs - Iabab.vovv
    tempab = einsum('ic,cjab->ijab',L1aold,ITvOvVs)
    ITvOvVs = None
    ITVovVs = IVovVs + Iabab.ovvv.transpose((1,0,2,3))
    tempab -= einsum('jc,ciab->ijab',L1bold,ITVovVs)
    ITVovVs = None
    L2ab -= fac*tempab

    ## OOOV
    ITooovs = Iooovs + Ia.ooov
    tempaa = einsum('ka,ijkb->ijab',L1aold,ITooovs)
    tempaa -= tempaa.transpose((0,1,3,2))
    ITooovs = None
    ITOOOVs = IOOOVs + Ib.ooov
    tempbb = einsum('ka,ijkb->ijab',L1bold,ITOOOVs)
    tempbb -= tempbb.transpose((0,1,3,2))
    ITOOOVs = None
    IToOoVs = IoOoVs + Iabab.ooov
    tempab = einsum('ka,ijkb->ijab',L1aold,IToOoVs)
    IToOoVs = None
    IToOOvs = IoOOvs - Iabab.oovo.transpose((0,1,3,2))
    tempab -= einsum('kb,ijka->ijab',L1bold,IToOOvs)
    IToOOvs = None
    L2aa -= fac*tempaa
    L2ab -= fac*tempab
    L2bb -= fac*tempbb

    ## VOVO
    ITvovos = -Ivovos1 - Ivovos2 - Ia.vovo
    L1a += fac*einsum('jb,biaj->ia',L1aold,ITvovos)
    ITvovos -= einsum('djcb,ck->djbk',Ivovvs,T1aold)
    tempaa = einsum('ikad,djbk->ijab',L2aaold,ITvovos)
    tempab = einsum('kjdb,diak->ijab',L2abold,ITvovos)
    ITvovos = None
    ITVOVOs = -IVOVOs1 - IVOVOs2 - Ib.vovo
    L1b += fac*einsum('jb,biaj->ia',L1bold,ITVOVOs)
    ITVOVOs -= einsum('djcb,ck->djbk',IVOVVs,T1bold)
    tempab += einsum('ikad,djbk->ijab',L2abold,ITVOVOs)
    tempbb = einsum('ikad,djbk->ijab',L2bbold,ITVOVOs)
    ITVOVOs = None
    ITvOVos = -IvOVos1 - IvOVos2 + Iabab.voov.transpose((0,1,3,2))
    L1b += fac*einsum('jb,biaj->ia',L1aold,ITvOVos)
    ITvOVos -= einsum('djcb,ck->djbk',IvOvVs,T1aold)
    tempab += einsum('ikad,djbk->ijab',L2aaold,ITvOVos)
    tempbb += einsum('kida,djbk->ijab',L2abold,ITvOVos)
    ITvOVos = None
    ITVovOs = -IVovOs1 - IVovOs2 + Iabab.ovvo.transpose((1,0,2,3))
    L1a += fac*einsum('jb,biaj->ia',L1bold,ITVovOs)
    ITVovOs -= einsum('djcb,ck->djbk',IVoVvs,T1bold)
    tempaa += einsum('ikad,djbk->ijab',L2abold,ITVovOs)
    tempab += einsum('kjdb,diak->ijab',L2bbold,ITVovOs)
    ITVovOs = None
    ITvOvOs = -IvOvOs1 - IvOvOs2 - Iabab.vovo
    ITvOvOs -= einsum('djcb,ck->djbk',IvOVvs,T1bold)
    tempab += einsum('ikdb,djak->ijab',L2abold,ITvOvOs)
    ITvOvOs = None
    ITVoVos = -IVoVos1 - IVoVos2 - Iabab.ovov.transpose((1,0,3,2))
    ITVoVos -= einsum('djcb,ck->djbk',IVovVs,T1aold)
    tempab += einsum('kjad,dibk->ijab',L2abold,ITVoVos)
    ITVoVos = None
    tempaa += tempaa.transpose((1,0,3,2)) - tempaa.transpose((0,1,3,2)) - tempaa.transpose((1,0,2,3))
    tempbb += tempbb.transpose((1,0,3,2)) - tempbb.transpose((0,1,3,2)) - tempbb.transpose((1,0,2,3))
    L2aa += fac*tempaa
    L2ab += fac*tempab
    L2bb += fac*tempbb

    ## VVVV
    ITvvvvs = einsum('dkab,ck->cdab',Ia.vovv,T1aold)\
        + 0.5*einsum('klab,cdkl->cdab',Ia.oovv,TTempaa)\
        + 0.5*Ia.vvvv
    L2aa += fac*einsum('ijcd,cdab->ijab',L2aaold,ITvvvvs)
    ITvvvvs = None
    ITVVVVs = einsum('dkab,ck->cdab',Ib.vovv,T1bold)\
        + 0.5*einsum('klab,cdkl->cdab',Ib.oovv,TTempbb)\
        + 0.5*Ib.vvvv
    L2bb += fac*einsum('ijcd,cdab->ijab',L2bbold,ITVVVVs)
    ITVVVVs = None
    ITvVvVs = -einsum('kdab,ck->cdab',Iabab.ovvv,T1aold)
    ITvVvVs += 0.5*einsum('klab,cdkl->cdab',Iabab.oovv,TTempab)
    ITvVvVs += 0.25*einsum('lkab,cdlk->cdab',Iabab.oovv,T2abold)
    ITvVvVs += 0.5*Iabab.vvvv
    L2ab += fac*einsum('ijcd,cdab->ijab',L2abold,ITvVvVs)
    ITvVvVs = None
    ITVvvVs = einsum('dkab,ck->cdab',Iabab.vovv,T1bold)
    ITVvvVs -= 0.5*einsum('lkab,dclk->cdab',Iabab.oovv,TTempab)
    ITVvvVs -= 0.25*einsum('klab,dckl->cdab',Iabab.oovv,T2abold)
    ITVvvVs -= 0.5*Iabab.vvvv.transpose((1,0,2,3))
    L2ab -= fac*einsum('ijdc,cdab->ijab',L2abold,ITVvvVs)
    ITVvvVs = None

    ## OOOO
    IToooos = einsum('ijlc,ck->ijkl',Ia.ooov,T1aold)\
        - 0.5*einsum('ijcd,cdkl->ijkl',Ia.oovv,TTempaa)\
        - 0.5*Ia.oooo
    L2aa -= fac*einsum('klab,ijkl->ijab',L2aaold,IToooos)
    IToooos = None
    ITOOOOs = einsum('ijlc,ck->ijkl',Ib.ooov,T1bold)\
        - 0.5*einsum('ijcd,cdkl->ijkl',Ib.oovv,TTempbb)\
        - 0.5*Ib.oooo
    L2bb -= fac*einsum('klab,ijkl->ijab',L2bbold,ITOOOOs)
    ITOOOOs = None
    IToOoOs = -einsum('ijcl,ck->ijkl',Iabab.oovo,T1aold)\
        - 0.5*einsum('ijcd,cdkl->ijkl',Iabab.oovv,TTempab)\
        - 0.25*einsum('ijdc,dckl->ijkl',Iabab.oovv,T2abold)
    IToOoOs += - 0.5*Iabab.oooo
    L2ab -= fac*einsum('klab,ijkl->ijab',L2abold,IToOoOs)
    IToOoOs = None
    IToOOos = einsum('ijlc,ck->ijkl',Iabab.ooov,T1bold)
    IToOOos += 0.5*einsum('ijdc,dclk->ijkl',Iabab.oovv,TTempab)
    IToOOos += 0.25*einsum('ijcd,cdlk->ijkl',Iabab.oovv,T2abold)
    IToOOos += 0.5*Iabab.oooo.transpose((0,1,3,2))
    L2ab += fac*einsum('lkab,ijkl->ijab',L2abold,IToOOos)
    IToOOos = None

    ## OOVV
    IToovvs = einsum('ljdb,cdkl->kjcb',Ia.oovv,T2aaold)
    IToovvs += einsum('jlbd,cdkl->kjcb',Iabab.oovv,T2abold)
    tempaa = einsum('ikac,kjcb->ijab',L2aaold,IToovvs)
    tempab = einsum('kjcb,kica->ijab',L2abold,IToovvs)
    IToovvs = None
    ITOOVVs = einsum('ljdb,cdkl->kjcb',Ib.oovv,T2bbold)
    ITOOVVs += einsum('ljdb,dclk->kjcb',Iabab.oovv,T2abold)
    tempbb = einsum('ikac,kjcb->ijab',L2bbold,ITOOVVs)
    tempab += einsum('ikac,kjcb->ijab',L2abold,ITOOVVs)
    ITOOVVs = None
    IToOvVs = einsum('ljdb,cdkl->kjcb',Iabab.oovv,T2aaold)
    IToOvVs += einsum('ljdb,cdkl->kjcb',Ib.oovv,T2abold)
    tempbb += einsum('kica,kjcb->ijab',L2abold,IToOvVs)
    tempab += einsum('ikac,kjcb->ijab',L2aaold,IToOvVs)
    IToOvVs = None
    ITOoVvs = einsum('ljdb,dclk->kjcb',Ia.oovv,T2abold)
    ITOoVvs += einsum('jlbd,cdkl->kjcb',Iabab.oovv,T2bbold)
    tempaa += einsum('ikac,kjcb->ijab',L2abold,ITOoVvs)
    tempab += einsum('kjcb,kica->ijab',L2bbold,ITOoVvs)
    ITOoVvs = None
    ITOOvvs = einsum('ljbd,cdlk->kjcb',Iabab.oovv,T2abold)
    tempab += einsum('ikcb,kjca->ijab',L2abold,ITOOvvs)
    ITOOvvs = None
    ITooVVs = einsum('jldb,dckl->kjcb',Iabab.oovv,T2abold)
    tempab += einsum('kjac,kicb->ijab',L2abold,ITooVVs)
    ITooVVs = None
    tempaa += tempaa.transpose((1,0,3,2)) - tempaa.transpose((0,1,3,2)) - tempaa.transpose((1,0,2,3))
    tempbb += tempbb.transpose((1,0,3,2)) - tempbb.transpose((0,1,3,2)) - tempbb.transpose((1,0,2,3))
    L2aa += fac*tempaa
    L2ab += fac*tempab
    L2bb += fac*tempbb

    ## LT terms
    Lt1a = einsum('jkbd,bcjk->cd',L2aaold,T2aaold)
    Lt1a += einsum('kjdb,cbkj->cd',L2abold,T2abold)
    Lt1a += einsum('jkdb,cbjk->cd',L2abold,T2abold)
    L1a += 0.5*fac*einsum('cd,dica->ia',Lt1a,Ia.vovv)
    L1b += 0.5*fac*einsum('cd,dica->ia',Lt1a,Iabab.vovv)
    L1a -= 0.5*fac*einsum('db,bida->ia',Lt1a,Ivovvs)
    L1b -= 0.5*fac*einsum('db,bida->ia',Lt1a,IvOvVs)

    Lt1b = einsum('jkbd,bcjk->cd',L2bbold,T2bbold)
    Lt1b += einsum('jkbd,bcjk->cd',L2abold,T2abold)
    Lt1b += einsum('kjbd,bckj->cd',L2abold,T2abold)
    L1a += 0.5*fac*einsum('cd,idac->ia',Lt1b,Iabab.ovvv)
    L1b += 0.5*fac*einsum('cd,dica->ia',Lt1b,Ib.vovv)
    L1a -= 0.5*fac*einsum('db,bida->ia',Lt1b,IVoVvs)
    L1b -= 0.5*fac*einsum('db,bida->ia',Lt1b,IVOVVs)

    Lt2a = einsum('jlbc,bcjk->lk',L2aaold,T2aaold)
    Lt2a += einsum('ljcb,cbkj->lk',L2abold,T2abold)
    Lt2a += einsum('ljbc,bckj->lk',L2abold,T2abold)
    L1a -= 0.5*fac*einsum('lk,kila->ia',Lt2a,Ia.ooov)
    L1b -= 0.5*fac*einsum('lk,kila->ia',Lt2a,Iabab.ooov)
    L1a -= 0.5*fac*einsum('jl,lija->ia',Lt2a,Iooovs)
    L1b -= 0.5*fac*einsum('jl,lija->ia',Lt2a,IoOoVs)

    Lt2b = einsum('jlbc,bcjk->lk',L2bbold,T2bbold)
    Lt2b += einsum('jlbc,bcjk->lk',L2abold,T2abold)
    Lt2b += einsum('jlcb,cbjk->lk',L2abold,T2abold)
    L1a -= 0.5*fac*einsum('lk,ikal->ia',Lt2b,Iabab.oovo)
    L1b -= 0.5*fac*einsum('lk,kila->ia',Lt2b,Ib.ooov)
    L1a -= 0.5*fac*einsum('jl,lija->ia',Lt2b,IOoOvs)
    L1b -= 0.5*fac*einsum('jl,lija->ia',Lt2b,IOOOVs)

    tempaa = 0.5*einsum('da,ijdb->ijab',Lt1a,Ia.oovv)
    tempab = 0.5*einsum('da,ijdb->ijab',Lt1a,Iabab.oovv)
    tempaa -= tempaa.transpose((0,1,3,2))
    tempbb = 0.5*einsum('da,ijdb->ijab',Lt1b,Ib.oovv)
    tempbb -= tempbb.transpose((0,1,3,2))
    tempab += 0.5*einsum('db,ijad->ijab',Lt1b,Iabab.oovv)
    L2aa -= fac*tempaa
    L2ab -= fac*tempab
    L2bb -= fac*tempbb

    tempaa = 0.5*einsum('il,ljab->ijab',Lt2a,Ia.oovv)
    tempaa -= tempaa.transpose((1,0,2,3))
    tempbb = 0.5*einsum('il,ljab->ijab',Lt2b,Ib.oovv)
    tempbb -= tempbb.transpose((1,0,2,3))
    tempab = 0.5*einsum('il,ljab->ijab',Lt2a,Iabab.oovv)
    tempab += 0.5*einsum('jl,ilab->ijab',Lt2b,Iabab.oovv)
    L2aa -= fac*tempaa
    L2ab -= fac*tempab
    L2bb -= fac*tempbb


def ccd_lambda_simple(F, I, L2old, T2old):
    """Coupled cluster doubles (CCD) Lambda iteration."""
    L2 = I.oovv.copy()

    _LD_LD(L2, F, I, L2old)
    _LD_LDTD(L2, I, L2old, T2old)

    return L2

def ccsd_lambda_opt(F, I, L1old, L2old, T1old, T2old):
    """Coupled cluster singles and doubles (CCSD) Lambda iteration
    with intermediats.
    """
    L1 = F.ov.copy()
    L2 = I.oovv.copy()

    _LS_TS(L1, I, T1old)
    _Lambda_opt(L1, L2, F, I, L1old, L2old, T1old, T2old)

    return L1,L2

def ccsd_lambda_opt_int(F, I, L1old, L2old, T1old, T2old, intor):
    """Coupled cluster singles and doubles (CCSD) Lambda iteration
    with intermediats.
    """
    L1 = F.ov.copy()
    L2 = I.oovv.copy()

    _LS_TS(L1, I, T1old)
    _Lambda_opt_int(L1, L2, F, I, L1old, L2old, T1old, T2old, intor)

    return L1,L2

def uccsd_lambda_opt(Fa, Fb, Ia, Ib, Iabab, L1old, L2old, T1old, T2old):
    """Coupled cluster singles and doubles (CCSD) Lambda iteration
    with intermediats.
    """
    # unpack
    L1aold,L1bold = L1old
    L2aaold,L2abold,L2bbold = L2old
    T1aold,T1bold = T1old
    T2aaold,T2abold,T2bbold = T2old

    # dims
    noa,nva = L1aold.shape
    nob,nvb = L1bold.shape

    L1a = Fa.ov.copy()
    L1b = Fb.ov.copy()
    L2aa = Ia.oovv.copy()
    L2ab = Iabab.oovv.copy()
    L2bb = Ib.oovv.copy()
    _u_LS_TS(L1a,L1b,Ia,Ib,Iabab,T1aold,T1bold)
    _uccsd_Lambda_opt(L1a, L1b, L2aa, L2ab, L2bb, Fa, Fb, Ia, Ib, Iabab, L1old, L2old, T1old, T2old)

    return (L1a,L1b),(L2aa,L2ab,L2bb)

def ccsd_lambda_simple(F, I, L1old, L2old, T1old, T2old):
    """Coupled cluster singles and doubles (CCSD) Lambda iteration."""
    L1 = F.ov.copy()
    L2 = I.oovv.copy()

    _LS_TS(L1, I, T1old)
    _LS_LS(L1, F, I, L1old)
    _LS_LSTS(L1, F, I, L1old, T1old)
    _LS_LSTD(L1, I, L1old, T2old)
    _LS_LSTSS(L1, I, L1old, T1old)
    _LS_LD(L1, F, I, L2old)
    _LS_LDTS(L1, F, I, L2old, T1old)
    _LS_LDTD(L1, F, I, L2old, T2old)
    _LS_LDTSS(L1, F, I, L2old, T1old)
    _LS_LDTSD(L1, I, L2old, T1old, T2old)
    _LS_LDTSSS(L1, I, L2old, T1old)

    _LD_LS(L2, F, I, L1old)
    _LD_LSTS(L2, F, I, L1old, T1old)
    _LD_LD(L2, F, I, L2old)
    _LD_LDTS(L2, F, I, L2old, T1old)
    _LD_LDTD(L2, I, L2old, T2old)
    _LD_LDTSS(L2, F, I, L2old, T1old)

    return L1,L2

def ccsd_1rdm_ba(T1,T2,L1,L2):
    pba = numpy.einsum('ia,bi->ba',L1,T1) \
        + 0.5*numpy.einsum('kica,cbki->ba',L2,T2)
    return pba

ccsd_1rdm_ba_opt = ccsd_1rdm_ba

def ccsd_1rdm_ji(T1,T2,L1,L2):
    pji = -numpy.einsum('ja,ai->ji',L1,T1) \
        - 0.5*numpy.einsum('kjca,caki->ji',L2,T2)
    return pji

ccsd_1rdm_ji_opt = ccsd_1rdm_ji

def ccsd_1rdm_ai(T1,T2,L1,L2,tfac=1.0):
    pai = tfac*T1\
        + numpy.einsum('jb,baji->ai',L1,T2)\
        - numpy.einsum('jb,bi,aj->ai',L1,T1,T1)\
        - 0.5*numpy.einsum('kjcb,ci,abkj->ai',L2,T1,T2)\
        - 0.5*numpy.einsum('kjcb,ak,cbij->ai',L2,T1,T2)
    return pai

def ccsd_2rdm_cdab(T1,T2,L1,L2):
    Pcdab = 0.5*einsum('ijab,cdij->cdab',L2,T2)
    Pcdab += 0.5*einsum('ijab,ci,dj->cdab',L2,T1,T1)
    Pcdab -= 0.5*einsum('ijab,di,cj->cdab',L2,T1,T1)
    return Pcdab

def ccsd_2rdm_ciab(T1,T2,L1,L2):
    Pciab = einsum('jiab,cj->ciab',L2,T1)
    return Pciab

ccsd_2rdm_ciab_opt = ccsd_2rdm_ciab

def ccsd_2rdm_bcai(T1,T2,L1,L2):
    Pbcai = einsum('ja,bcji->bcai',L1,T2)
    Pbcai += einsum('ja,bj,ci->bcai',L1,T1,T1)
    Pbcai -= einsum('ja,cj,bi->bcai',L1,T1,T1)
    Pbcai += 0.5*einsum('jlad,bdjl,ci->bcai',L2,T2,T1)
    Pbcai -= 0.5*einsum('jlad,cdjl,bi->bcai',L2,T2,T1)
    Pbcai += einsum('jkad,cdik,bj->bcai',L2,T2,T1)
    Pbcai -= einsum('jkad,bdik,cj->bcai',L2,T2,T1)
    Pbcai -= 0.5*einsum('jkda,cbjk,di->bcai',L2,T2,T1)
    Pbcai -= einsum('kjda,ck,di,bj->bcai',L2,T1,T1,T1)
    return Pbcai

def ccsd_2rdm_bjai(T1,T2,L1,L2):
    Pbjai = -einsum('ja,bi->bjai',L1,T1)
    Pbjai -= einsum('kjac,bcki->bjai',L2,T2)
    Pbjai -= einsum('kjac,bk,ci->bjai',L2,T1,T1)
    return Pbjai

def ccsd_2rdm_abij(T1,T2,L1,L2,tfac=1.0):
    Pabij = tfac*T2.copy()
    Pabij += tfac*einsum('ai,bj->abij',T1,T1)
    Pabij -= tfac*einsum('aj,bi->abij',T1,T1)
    Pabij -= einsum('kc,abkj,ci->abij',L1,T2,T1)
    Pabij += einsum('kc,abki,cj->abij',L1,T2,T1)
    Pabij -= einsum('kc,cbij,ak->abij',L1,T2,T1)
    Pabij += einsum('kc,caij,bk->abij',L1,T2,T1)
    Pabij += einsum('kc,bcjk,ai->abij',L1,T2,T1)
    Pabij -= einsum('kc,acjk,bi->abij',L1,T2,T1)
    Pabij -= einsum('kc,bcik,aj->abij',L1,T2,T1)
    Pabij += einsum('kc,acik,bj->abij',L1,T2,T1)
    Pabij -= einsum('kc,ak,bj,ci->abij',L1,T1,T1,T1)
    Pabij += einsum('kc,bk,aj,ci->abij',L1,T1,T1,T1)
    Pabij += einsum('kc,ak,bi,cj->abij',L1,T1,T1,T1)
    Pabij -= einsum('kc,bk,ai,cj->abij',L1,T1,T1,T1)
    Pabij += 0.25*einsum('klcd,abkl,cdij->abij',L2,T2,T2)
    Pabij += 0.5*einsum('klcd,caki,bdjl->abij',L2,T2,T2)
    Pabij -= 0.5*einsum('klcd,cakj,bdil->abij',L2,T2,T2)
    Pabij -= 0.5*einsum('klcd,cbki,adjl->abij',L2,T2,T2)
    Pabij += 0.5*einsum('klcd,cbkj,adil->abij',L2,T2,T2)
    Pabij -= 0.5*einsum('klcd,bdkl,acij->abij',L2,T2,T2)
    Pabij += 0.5*einsum('klcd,adkl,bcij->abij',L2,T2,T2)
    Pabij -= 0.5*einsum('klcd,abik,cdjl->abij',L2,T2,T2)
    Pabij += 0.5*einsum('klcd,abjk,cdil->abij',L2,T2,T2)
    Pabij += 0.25*einsum('klcd,cdij,ak,bl->abij',L2,T2,T1,T1)
    Pabij -= 0.25*einsum('klcd,cdij,bk,al->abij',L2,T2,T1,T1)
    Pabij += 0.25*einsum('klcd,abkl,ci,dj->abij',L2,T2,T1,T1)
    Pabij -= 0.25*einsum('klcd,abkl,cj,di->abij',L2,T2,T1,T1)
    Pabij -= einsum('klcd,bdjl,ak,ci->abij',L2,T2,T1,T1)
    Pabij += einsum('klcd,adjl,bk,ci->abij',L2,T2,T1,T1)
    Pabij += einsum('klcd,bdil,ak,cj->abij',L2,T2,T1,T1)
    Pabij -= einsum('klcd,adil,bk,cj->abij',L2,T2,T1,T1)
    Pabij -= 0.5*einsum('klcd,cdjl,bk,ai->abij',L2,T2,T1,T1)
    Pabij += 0.5*einsum('klcd,cdjl,ak,bi->abij',L2,T2,T1,T1)
    Pabij += 0.5*einsum('klcd,cdil,bk,aj->abij',L2,T2,T1,T1)
    Pabij -= 0.5*einsum('klcd,cdil,ak,bj->abij',L2,T2,T1,T1)
    Pabij -= 0.5*einsum('klcd,bdkl,cj,ai->abij',L2,T2,T1,T1)
    Pabij += 0.5*einsum('klcd,adkl,cj,bi->abij',L2,T2,T1,T1)
    Pabij += 0.5*einsum('klcd,bdkl,ci,aj->abij',L2,T2,T1,T1)
    Pabij -= 0.5*einsum('klcd,adkl,ci,bj->abij',L2,T2,T1,T1)
    Pabij += 0.25*einsum('klcd,ci,ak,dj,bl->abij',L2,T1,T1,T1,T1)
    Pabij -= 0.25*einsum('klcd,ci,bk,dj,al->abij',L2,T1,T1,T1,T1)
    Pabij -= 0.25*einsum('klcd,cj,ak,di,bl->abij',L2,T1,T1,T1,T1)
    Pabij += 0.25*einsum('klcd,cj,bk,di,al->abij',L2,T1,T1,T1,T1)
    return Pabij

def ccsd_2rdm_jkai(T1,T2,L1,L2):
    Pjkai = -einsum('jkab,bi->jkai',L2,T1)
    return Pjkai

ccsd_2rdm_jkai_opt = ccsd_2rdm_jkai

def ccsd_2rdm_kaij(T1,T2,L1,L2):
    Pkaij = -einsum('kb,baij->kaij',L1,T2)
    Pkaij -= einsum('kb,bi,aj->kaij',L1,T1,T1)
    Pkaij += einsum('kb,bj,ai->kaij',L1,T1,T1)
    Pkaij -= 0.5*einsum('klcd,cdil,aj->kaij',L2,T2,T1)
    Pkaij += 0.5*einsum('klcd,cdjl,ai->kaij',L2,T2,T1)
    Pkaij -= einsum('klcd,adjl,ci->kaij',L2,T2,T1)
    Pkaij += einsum('klcd,adil,cj->kaij',L2,T2,T1)
    Pkaij += 0.5*einsum('lkdb,dbji,al->kaij',L2,T2,T1)
    Pkaij += 0.5*einsum('lkdb,al,dj,bi->kaij',L2,T1,T1,T1)
    Pkaij -= 0.5*einsum('lkdb,al,di,bj->kaij',L2,T1,T1,T1)
    return Pkaij

def ccsd_2rdm_klij(T1,T2,L1,L2):
    Pklij = 0.5*einsum('klab,abij->klij',L2,T2)
    Pklij += 0.5*einsum('klab,ai,bj->klij',L2,T1,T1)
    Pklij -= 0.5*einsum('klab,aj,bi->klij',L2,T1,T1)
    return Pklij

#def ccsd_1rdm_ba_opt(T1,T2,L1,L2):
#    pba = numpy.einsum('ia,bi->ba',L1,T1) \
#        + 0.5*numpy.einsum('kicb,caki->ab',L2,T2)
#    return pba

#def ccsd_1rdm_ji_opt(T1,T2,L1,L2):
#    pji = -numpy.einsum('ia,aj->ij',L1,T1) \
#        - 0.5*numpy.einsum('kiab,abkj->ij',L2,T2)
#    return pji

def ccsd_1rdm_ai_opt(T1,T2,L1,L2,tfac=1.0):
    pai = tfac*T1
    T2temp = T2 - einsum('bi,aj->baji',T1,T1)
    pai += numpy.einsum('jb,baji->ai',L1,T2temp)

    Pac = 0.5*einsum('kjcb,abkj->ac',L2,T2)
    pai -= einsum('ac,ci->ai',Pac,T1)

    Pik = 0.5*einsum('kjcb,cbij->ik',L2,T2)
    pai -= einsum('ik,ak->ai',Pik,T1)
    return pai

def ccsd_2rdm_cdab_opt(T1,T2,L1,L2):
    T2temp = T2 + einsum('ci,dj->cdij',T1,T1)
    T2temp -= einsum('di,cj->cdij',T1,T1)
    Pcdab = 0.5*einsum('ijab,cdij->cdab',L2,T2temp)
    return Pcdab

def ccsd_2rdm_bcai_opt(T1,T2,L1,L2):
    T2temp = T2 + einsum('ci,dj->cdij',T1,T1)
    T2temp -= einsum('di,cj->cdij',T1,T1)
    Pbcai = einsum('ja,bcji->bcai',L1,T2temp)

    LTba = einsum('jlad,bdjl->ba',L2,T2)
    Pbcai += 0.5*einsum('ba,ci->bcai',LTba,T1)
    Pbcai -= 0.5*einsum('ca,bi->bcai',LTba,T1)

    LTtemp = einsum('jkad,cdik->jcai',L2,T2)
    Pbcai += einsum('jcai,bj->bcai',LTtemp,T1)
    Pbcai -= einsum('jbai,cj->bcai',LTtemp,T1)

    Pbcai -= 0.5*einsum('kjda,cbkj,di->bcai',L2,T2temp,T1)
    return Pbcai

def ccsd_2rdm_bjai_opt(T1,T2,L1,L2):
    Pbjai = -einsum('ja,bi->bjai',L1,T1)
    T2temp = T2 + einsum('bk,ci->bcki',T1,T1)
    Pbjai -= einsum('kjac,bcki->bjai',L2,T2temp)
    return Pbjai

def ccsd_2rdm_abij_opt(T1,T2,L1,L2,tfac=1.0):
    Pabij = tfac*T2.copy()
    Pabij += tfac*einsum('ai,bj->abij',T1,T1)
    Pabij -= tfac*einsum('aj,bi->abij',T1,T1)

    LTki = einsum('kc,ci->ki',L1,T1)
    tmp = -einsum('ki,abkj->abij',LTki,T2)
    Pabij += tmp - tmp.transpose((0,1,3,2))

    LTac = einsum('kc,ak->ac',L1,T1)
    tmp = -einsum('ac,cbij->abij',LTac,T2)
    Pabij += tmp - tmp.transpose((1,0,2,3))

    T2temp = T2 - einsum('bk,cj->bcjk',T1,T1)
    LTbj = einsum('kc,bcjk->bj',L1,T2temp)
    Pabij += einsum('ai,bj->abij',LTbj,T1)
    Pabij -= einsum('aj,bi->abij',LTbj,T1)
    Pabij -= einsum('bi,aj->abij',LTbj,T1)
    Pabij += einsum('bj,ai->abij',LTbj,T1)

    LToo = einsum('klcd,cdij->klij',L2,T2)
    Pabij += 0.25*einsum('klij,abkl->abij',LToo,T2)

    LTov = einsum('klcd,caki->lida',L2,T2)
    tmp = 0.5*einsum('lida,bdjl->abij',LTov,T2)
    Pabij += tmp - tmp.transpose((0,1,3,2)) - tmp.transpose((1,0,2,3)) + tmp.transpose((1,0,3,2))

    T2temp = T2 + einsum('cj,ai->acij',T1,T1) - einsum('ci,aj->acij',T1,T1)
    Lcb = einsum('klcd,bdkl->cb',L2,T2)
    tmp = -0.5*einsum('cb,acij->abij',Lcb,T2temp)
    Pabij += tmp - tmp.transpose((1,0,2,3))

    Lkj = einsum('klcd,cdjl->kj',L2,T2)
    tmp = -0.5*einsum('kj,abik->abij',Lkj,T2temp)
    Pabij += tmp - tmp.transpose((0,1,3,2))

    T2temp = T2 + einsum('ci,dj->cdij',T1,T1)
    LToo = einsum('klcd,cdij->klij',L2,T2temp)
    tmp = einsum('klij,ak->alij',LToo,T1)
    tmp = 0.25*einsum('alij,bl->abij',tmp,T1)
    Pabij += tmp - tmp.transpose((1,0,2,3))

    Looov = einsum('klcd,ci->klid',L2,T1)
    Loooo = einsum('klid,dj->klij',Looov,T1)
    Loooo -= Loooo.transpose((0,1,3,2))
    Pabij += 0.25*einsum('klij,abkl->abij',Loooo,T2temp)

    Lalid = einsum('klcd,ak,ci->alid',L2,T1,T1)
    tmp = einsum('alid,bdjl->abij',Lalid,T2)
    Pabij -= tmp
    Pabij += tmp.transpose((0,1,3,2)) + tmp.transpose((1,0,2,3))
    Pabij -= tmp.transpose((1,0,3,2))
    return Pabij

def ccsd_2rdm_kaij_opt(T1,T2,L1,L2):
    T2temp = T2 + einsum('ci,dj->cdij',T1,T1)
    T2temp -= einsum('di,cj->cdij',T1,T1)
    Pkaij = -einsum('kb,baij->kaij',L1,T2temp)

    LTo = einsum('klcd,cdil->ki',L2,T2)
    tmp = -0.5*einsum('ki,aj->kaij',LTo,T1)
    Pkaij += tmp - tmp.transpose((0,1,3,2))

    Lklid = einsum('klcd,ci->klid',L2,T1)
    tmp = -einsum('klid,adjl->kaij',Lklid,T2)
    Pkaij += tmp - tmp.transpose((0,1,3,2))

    Pkaij += 0.5*einsum('lkdb,dbji,al->kaij',L2,T2temp,T1)
    return Pkaij

def ccsd_2rdm_klij_opt(T1,T2,L1,L2):
    T2temp = T2 + einsum('ci,dj->cdij',T1,T1)
    T2temp -= einsum('di,cj->cdij',T1,T1)
    Pklij = 0.5*einsum('klab,abij->klij',L2,T2temp)
    return Pklij

def uccsd_1rdm_ba(T1a,T1b,T2aa,T2ab,T2bb,
        L1a,L1b,L2aa,L2ab,L2bb):
    pba_a = numpy.einsum('ia,bi->ba',L1a,T1a)
    pba_a += 0.5*numpy.einsum('kica,cbki->ba',L2aa,T2aa)
    pba_a += numpy.einsum('ikac,bcik->ba',L2ab,T2ab)

    pba_b = numpy.einsum('ia,bi->ba',L1b,T1b)
    pba_b += 0.5*numpy.einsum('kica,cbki->ba',L2bb,T2bb)
    pba_b += numpy.einsum('kica,cbki->ba',L2ab,T2ab)
    return pba_a,pba_b

def uccsd_1rdm_ji(T1a,T1b,T2aa,T2ab,T2bb,
        L1a,L1b,L2aa,L2ab,L2bb):
    pji_a = -numpy.einsum('ja,ai->ji',L1a,T1a)
    pji_a -= 0.5*numpy.einsum('kjca,caki->ji',L2aa,T2aa)
    pji_a -= numpy.einsum('jkac,acik->ji',L2ab,T2ab)

    pji_b = -numpy.einsum('ja,ai->ji',L1b,T1b)
    pji_b -= 0.5*numpy.einsum('kjca,caki->ji',L2bb,T2bb)
    pji_b -= numpy.einsum('kjca,caki->ji',L2ab,T2ab)

    return pji_a, pji_b

def uccsd_1rdm_ai(T1a,T1b,T2aa,T2ab,T2bb,
        L1a,L1b,L2aa,L2ab,L2bb,tfac=1.0):
    T2tempaa = T2aa - einsum('bi,aj->baji',T1a,T1a)
    T2tempbb = T2bb - einsum('bi,aj->baji',T1b,T1b)
    T2tempab = T2ab

    pai_a = tfac*T1a
    pai_a += numpy.einsum('jb,baji->ai',L1a,T2tempaa)
    pai_a += numpy.einsum('jb,abij->ai',L1b,T2tempab)

    Pac_a = 0.5*einsum('kjcb,abkj->ac',L2aa,T2aa)
    Pac_a += einsum('kjcb,abkj->ac',L2ab,T2ab)
    pai_a -= einsum('ac,ci->ai',Pac_a,T1a)

    Pik_a = 0.5*einsum('kjcb,cbij->ik',L2aa,T2aa)
    Pik_a += einsum('kjcb,cbij->ik',L2ab,T2ab)
    pai_a -= einsum('ik,ak->ai',Pik_a,T1a)

    pai_b = tfac*T1b
    pai_b += numpy.einsum('jb,baji->ai',L1b,T2tempbb)
    pai_b += numpy.einsum('jb,baji->ai',L1a,T2tempab)

    Pac_b = 0.5*einsum('kjcb,abkj->ac',L2bb,T2bb)
    Pac_b += einsum('jkbc,bajk->ac',L2ab,T2ab)
    pai_b -= einsum('ac,ci->ai',Pac_b,T1b)

    Pik_b = 0.5*einsum('kjcb,cbij->ik',L2bb,T2bb)
    Pik_b += einsum('jkbc,bcji->ik',L2ab,T2ab)
    pai_b -= einsum('ik,ak->ai',Pik_b,T1b)
    return pai_a,pai_b

def uccsd_2rdm_ciab(T1a,T1b,T2aa,T2ab,T2bb,
        L1a,L1b,L2aa,L2ab,L2bb):
    Pciab = einsum('jiab,cj->ciab',L2aa,T1a)
    PCIAB = einsum('jiab,cj->ciab',L2bb,T1b)
    PcIaB = einsum('jiab,cj->ciab',L2ab,T1a)
    PCiAb = einsum('ijba,cj->ciab',L2ab,T1b)

    return Pciab,PCIAB,PcIaB,PCiAb

def uccsd_2rdm_jkai(T1a,T1b,T2aa,T2ab,T2bb,
        L1a,L1b,L2aa,L2ab,L2bb):
    Pjkai = -einsum('jkab,bi->jkai',L2aa,T1a)
    PjKaI = -einsum('jkab,bi->jkai',L2ab,T1b)
    PJkAi = -einsum('kJbA,bi->JkAi',L2ab,T1a)
    PJKAI = -einsum('jkab,bi->jkai',L2bb,T1b)
    return Pjkai,PJKAI,PjKaI,PJkAi

def uccsd_2rdm_cdab(T1a,T1b,T2aa,T2ab,T2bb,
        L1a,L1b,L2aa,L2ab,L2bb):
    T2tempaa = T2aa + einsum('ci,dj->cdij',T1a,T1a)
    T2tempbb = T2bb + einsum('ci,dj->cdij',T1b,T1b)
    T2tempab = T2ab + einsum('ci,dj->cdij',T1a,T1b)
    T2tempaa -= einsum('di,cj->cdij',T1a,T1a)
    T2tempbb -= einsum('di,cj->cdij',T1b,T1b)
    Pcdab = 0.5*einsum('ijab,cdij->cdab',L2aa,T2tempaa)
    PCDAB = 0.5*einsum('ijab,cdij->cdab',L2bb,T2tempbb)
    PcDaB = einsum('ijab,cdij->cdab',L2ab,T2tempab)
    return Pcdab,PCDAB,PcDaB

def uccsd_2rdm_bjai(T1a,T1b,T2aa,T2ab,T2bb,
        L1a,L1b,L2aa,L2ab,L2bb):
    Pbjai = -einsum('bi,ja->bjai',T1a,L1a)
    PbJAi = -einsum('bi,ja->bjai',T1a,L1b)
    PBjaI = -einsum('bi,ja->bjai',T1b,L1a)
    PBJAI = -einsum('bi,ja->bjai',T1b,L1b)
    T2tempaa = T2aa + einsum('bk,ci->bcki',T1a,T1a)
    T2tempab = T2ab + einsum('bk,ci->bcki',T1a,T1b)
    T2tempbb = T2bb + einsum('bk,ci->bcki',T1b,T1b)
    Pbjai -= einsum('kjac,bcki->bjai',L2aa,T2tempaa)
    Pbjai -= einsum('jKaC,bCiK->bjai',L2ab,T2ab)

    PbJaI = -einsum('kJaC,bCkI->bJaI',L2ab,T2tempab)

    PbJAi += einsum('kJcA,bcki->bJAi',L2ab,T2tempaa)
    PbJAi += einsum('KJAC,bCiK->bJAi',L2bb,T2ab)

    PBjaI += einsum('kjac,cBkI->BjaI',L2aa,T2ab)
    PBjaI += einsum('jKaC,BCKI->BjaI',L2ab,T2tempbb)

    PBjAi = -einsum('jKcA,cBiK->BjAi',L2ab,T2tempab)

    PBJAI -= einsum('kjac,bcki->bjai',L2bb,T2tempbb)
    PBJAI -= einsum('kJcA,cBkI->BJAI',L2ab,T2ab)
    return Pbjai,PBJAI,PbJaI,PbJAi,PBjaI,PBjAi

def uccsd_2rdm_klij(T1a,T1b,T2aa,T2ab,T2bb,
        L1a,L1b,L2aa,L2ab,L2bb):
    T2tempaa = T2aa + einsum('ci,dj->cdij',T1a,T1a)
    T2tempab = T2ab + einsum('ci,dj->cdij',T1a,T1b)
    T2tempbb = T2bb + einsum('ci,dj->cdij',T1b,T1b)
    T2tempaa -= einsum('di,cj->cdij',T1a,T1a)
    T2tempbb -= einsum('di,cj->cdij',T1b,T1b)
    Pklij = 0.5*einsum('klab,abij->klij',L2aa,T2tempaa)
    PkLiJ = einsum('klab,abij->klij',L2ab,T2tempab)
    PKLIJ = 0.5*einsum('klab,abij->klij',L2bb,T2tempbb)
    return Pklij,PKLIJ,PkLiJ

def uccsd_2rdm_bcai(T1a,T1b,T2aa,T2ab,T2bb,
        L1a,L1b,L2aa,L2ab,L2bb):
    T2tempaa = T2aa + einsum('ci,dj->cdij',T1a,T1a)
    T2tempab = T2ab + einsum('ci,dj->cdij',T1a,T1b)
    T2tempbb = T2bb + einsum('ci,dj->cdij',T1b,T1b)
    T2tempaa -= einsum('di,cj->cdij',T1a,T1a)
    T2tempbb -= einsum('di,cj->cdij',T1b,T1b)
    Pbcai = einsum('ja,bcji->bcai',L1a,T2tempaa)
    PbCaI = einsum('ja,bCjI->bCaI',L1a,T2tempab)
    PBcAi = einsum('JA,cBiJ->BcAi',L1b,T2tempab)
    PBCAI = einsum('ja,bcji->bcai',L1b,T2tempbb)

    LTba = einsum('jlad,bdjl->ba',L2aa,T2aa)
    LTba += 2.0*einsum('jLaD,bDjL->ba',L2ab,T2ab)
    LTBA = einsum('jlad,bdjl->ba',L2bb,T2bb)
    LTBA += 2.0*einsum('lJdA,dBlJ->BA',L2ab,T2ab)
    Pbcai += 0.5*einsum('ba,ci->bcai',LTba,T1a)
    Pbcai -= 0.5*einsum('ca,bi->bcai',LTba,T1a)

    PBCAI += 0.5*einsum('ba,ci->bcai',LTBA,T1b)
    PBCAI -= 0.5*einsum('ca,bi->bcai',LTBA,T1b)

    PbCaI += 0.5*einsum('ba,CI->bCaI',LTba,T1b)

    PBcAi += 0.5*einsum('BA,ci->BcAi',LTBA,T1a)

    LTtempaa = einsum('jkad,cdik->jcai',L2aa,T2aa)
    LTtempaa += einsum('jKaD,cDiK->jcai',L2ab,T2ab)
    LTtempbb = einsum('jkad,cdik->jcai',L2bb,T2bb)
    LTtempbb += einsum('kJdA,dCkI->JCAI',L2ab,T2ab)
    LTtempab1 = einsum('jkad,dCkI->jCaI',L2aa,T2ab)
    LTtempab1 += einsum('jKaD,CDIK->jCaI',L2ab,T2bb)
    LTtempab2 = einsum('kJaD,cDkI->JcaI',L2ab,T2ab)
    LTtempab3 = einsum('kJdA,cdik->JcAi',L2ab,T2aa)
    LTtempab3 += einsum('JKAD,cDiK->JcAi',L2bb,T2ab)
    LTtempab4 = einsum('jKdA,dCiK->jCAi',L2ab,T2ab)

    Pbcai += einsum('jcai,bj->bcai',LTtempaa,T1a)
    Pbcai -= einsum('jbai,cj->bcai',LTtempaa,T1a)

    PBCAI += einsum('jcai,bj->bcai',LTtempbb,T1b)
    PBCAI -= einsum('jbai,cj->bcai',LTtempbb,T1b)

    PbCaI += einsum('jCaI,bj->bCaI',LTtempab1,T1a)
    PbCaI -= einsum('JbaI,CJ->bCaI',LTtempab2,T1b)

    PBcAi += einsum('JcAi,BJ->BcAi',LTtempab3,T1b)
    PBcAi -= einsum('jBAi,cj->BcAi',LTtempab4,T1a)

    Pbcai -= 0.5*einsum('kjda,cbkj,di->bcai',L2aa,T2tempaa,T1a)
    PbCaI -= einsum('jKaD,bCjK,DI->bCaI',L2ab,T2tempab,T1b)
    PBcAi -= einsum('kJdA,cBkJ,di->BcAi',L2ab,T2tempab,T1a)
    PBCAI -= 0.5*einsum('kjda,cbkj,di->bcai',L2bb,T2tempbb,T1b)
    return Pbcai, PBCAI, PbCaI, PBcAi

def uccsd_2rdm_kaij(T1a,T1b,T2aa,T2ab,T2bb,
        L1a,L1b,L2aa,L2ab,L2bb):

    T2tempaa = T2aa + einsum('ci,dj->cdij',T1a,T1a)
    T2tempab = T2ab + einsum('ci,dj->cdij',T1a,T1b)
    T2tempbb = T2bb + einsum('ci,dj->cdij',T1b,T1b)
    T2tempaa -= einsum('di,cj->cdij',T1a,T1a)
    T2tempbb -= einsum('di,cj->cdij',T1b,T1b)
    Pkaij = -einsum('kb,baij->kaij',L1a,T2tempaa)
    PkAiJ = -einsum('kb,bAiJ->kAiJ',L1a,T2tempab)
    PKaIj = -einsum('KB,aBjI->KaIj',L1b,T2tempab)
    PKAIJ = -einsum('kb,baij->kaij',L1b,T2tempbb)

    LToa = einsum('klcd,cdil->ki',L2aa,T2aa)
    LToa += 2.0*einsum('kLcD,cDiL->ki',L2ab,T2ab)
    LTob = einsum('klcd,cdil->ki',L2bb,T2bb)
    LTob += 2.0*einsum('lKdC,dClI->KI',L2ab,T2ab)
    tmpaa = -0.5*einsum('ki,aj->kaij',LToa,T1a)
    tmpab = -0.5*einsum('ki,AJ->kAiJ',LToa,T1b)
    tmpba = -0.5*einsum('KI,aj->KaIj',LTob,T1a)
    tmpbb = -0.5*einsum('ki,aj->kaij',LTob,T1b)
    Pkaij += tmpaa - tmpaa.transpose((0,1,3,2))
    PkAiJ += tmpab
    PKaIj += tmpba
    PKAIJ += tmpbb - tmpbb.transpose((0,1,3,2))

    Lklid = einsum('klcd,ci->klid',L2aa,T1a)
    LkLiD = einsum('kLcD,ci->kLiD',L2ab,T1a)
    LKlId = einsum('lKdC,CI->KlId',L2ab,T1b)
    LKLID = einsum('klcd,ci->klid',L2bb,T1b)
    LkLId = -einsum('kLdC,CI->kLId',L2ab,T1b)
    LKliD = -einsum('lKcD,ci->KliD',L2ab,T1a)
    tmpaa = -einsum('klid,adjl->kaij',Lklid,T2aa)
    tmpaa -= einsum('kLiD,aDjL->kaij',LkLiD,T2ab)

    tmpab = -einsum('klid,dAlJ->kAiJ',Lklid,T2ab)
    tmpab -= einsum('kLiD,ADJL->kAiJ',LkLiD,T2bb)
    tmpab -= einsum('kLJd,dAiL->kAiJ',LkLId,T2ab)

    tmpba = -einsum('KlId,adjl->KaIj',LKlId,T2aa)
    tmpba -= einsum('KLID,aDjL->KaIj',LKLID,T2ab)
    tmpba -= einsum('KljD,aDlI->KaIj',LKliD,T2ab)

    tmpbb = -einsum('klid,adjl->kaij',LKLID,T2bb)
    tmpbb -= einsum('KlId,dAlJ->KAIJ',LKlId,T2ab)

    Pkaij += tmpaa - tmpaa.transpose((0,1,3,2))
    PkAiJ += tmpab
    PKaIj += tmpba
    PKAIJ += tmpbb - tmpbb.transpose((0,1,3,2))

    Pkaij += 0.5*einsum('lkdb,dbji,al->kaij',L2aa,T2tempaa,T1a)
    PkAiJ += einsum('kLbD,bDiJ,AL->kAiJ',L2ab,T2tempab,T1b)
    PKaIj += einsum('lKdB,dBjI,al->KaIj',L2ab,T2tempab,T1a)
    PKAIJ += 0.5*einsum('lkdb,dbji,al->kaij',L2bb,T2tempbb,T1b)
    return Pkaij,PKAIJ,PkAiJ,PKaIj

def uccsd_2rdm_abij(T1a,T1b,T2aa,T2ab,T2bb,
        L1a,L1b,L2aa,L2ab,L2bb,tfac=1.0):

    Pabij = tfac*T2aa.copy()
    PaBiJ = tfac*T2ab.copy()
    PABIJ = tfac*T2bb.copy()
    Pabij += tfac*einsum('ai,bj->abij',T1a,T1a)
    Pabij -= tfac*einsum('aj,bi->abij',T1a,T1a)
    PABIJ += tfac*einsum('ai,bj->abij',T1b,T1b)
    PABIJ -= tfac*einsum('aj,bi->abij',T1b,T1b)
    PaBiJ += tfac*einsum('ai,bj->abij',T1a,T1b)

    LTki = einsum('kc,ci->ki',L1a,T1a)
    LTKI = einsum('KC,CI->KI',L1b,T1b)
    tmpaa = -einsum('ki,abkj->abij',LTki,T2aa)
    tmpbb = -einsum('ki,abkj->abij',LTKI,T2bb)
    tmpab = -einsum('ki,aBkJ->aBiJ',LTki,T2ab)
    tmpab -= einsum('KJ,aBiK->aBiJ',LTKI,T2ab)
    Pabij += tmpaa - tmpaa.transpose((0,1,3,2))
    PABIJ += tmpbb - tmpbb.transpose((0,1,3,2))
    PaBiJ += tmpab

    LTac = einsum('kc,ak->ac',L1a,T1a)
    LTAC = einsum('kc,ak->ac',L1b,T1b)
    tmpaa = -einsum('ac,cbij->abij',LTac,T2aa)
    tmpbb = -einsum('ac,cbij->abij',LTAC,T2bb)
    tmpab = -einsum('ac,cBiJ->aBiJ',LTac,T2ab)
    tmpab -= einsum('BC,aCiJ->aBiJ',LTAC,T2ab)
    Pabij += tmpaa - tmpaa.transpose((1,0,2,3))
    PABIJ += tmpbb - tmpbb.transpose((1,0,2,3))
    PaBiJ += tmpab

    T2tempaa = T2aa - einsum('bk,cj->bcjk',T1a,T1a)
    T2tempbb = T2bb - einsum('bk,cj->bcjk',T1b,T1b)
    LTbj = einsum('kc,bcjk->bj',L1a,T2tempaa)
    LTbj += einsum('KC,bCjK->bj',L1b,T2ab)
    LTBJ = einsum('kc,bcjk->bj',L1b,T2tempbb)
    LTBJ += einsum('kc,cBkJ->BJ',L1a,T2ab)
    Pabij += einsum('ai,bj->abij',LTbj,T1a)
    Pabij -= einsum('aj,bi->abij',LTbj,T1a)
    Pabij -= einsum('bi,aj->abij',LTbj,T1a)
    Pabij += einsum('bj,ai->abij',LTbj,T1a)
    PABIJ += einsum('ai,bj->abij',LTBJ,T1b)
    PABIJ -= einsum('aj,bi->abij',LTBJ,T1b)
    PABIJ -= einsum('bi,aj->abij',LTBJ,T1b)
    PABIJ += einsum('bj,ai->abij',LTBJ,T1b)
    PaBiJ += einsum('ai,BJ->aBiJ',LTbj,T1b)
    PaBiJ += einsum('BJ,ai->aBiJ',LTBJ,T1a)

    LToo = einsum('klcd,cdij->klij',L2aa,T2aa)
    LToO = einsum('kLcD,cDiJ->kLiJ',L2ab,T2ab)
    LTOO = einsum('klcd,cdij->klij',L2bb,T2bb)
    Pabij += 0.25*einsum('klij,abkl->abij',LToo,T2aa)
    PABIJ += 0.25*einsum('klij,abkl->abij',LTOO,T2bb)
    PaBiJ += einsum('kLiJ,aBkL->aBiJ',LToO,T2ab)

    LTlida = einsum('klcd,caki->lida',L2aa,T2aa)
    LTlida += einsum('lKdC,aCiK->lida',L2ab,T2ab)
    LTLIDA = einsum('klcd,caki->lida',L2bb,T2bb)
    LTLIDA += einsum('kLcD,cAkI->LIDA',L2ab,T2ab)

    LTLIda = einsum('kLdC,aCkI->LIda',L2ab,T2ab)
    LTliDA = einsum('lKcD,cAiK->liDA',L2ab,T2ab)
    LTlIdA = einsum('klcd,cAkI->lIdA',L2aa,T2ab)
    LTlIdA += einsum('lKdC,CAKI->lIdA',L2ab,T2bb)
    LTLiDa = einsum('kLcD,caki->LiDa',L2ab,T2aa)
    LTLiDa += einsum('KLCD,aCiK->LiDa',L2bb,T2ab)

    tmpaa = 0.5*einsum('lida,bdjl->abij',LTlida,T2aa)
    tmpaa += 0.5*einsum('LiDa,bDjL->abij',LTLiDa,T2ab)
    tmpbb = 0.5*einsum('lida,bdjl->abij',LTLIDA,T2bb)
    tmpbb += 0.5*einsum('lIdA,dBlJ->ABIJ',LTlIdA,T2ab)

    tmpab = 0.5*einsum('lida,dBlJ->aBiJ',LTlida,T2ab)
    tmpab += 0.5*einsum('LiDa,BDJL->aBiJ',LTLiDa,T2bb)
    tmpab += 0.5*einsum('LJda,dBiL->aBiJ',LTLIda,T2ab)
    tmpab += 0.5*einsum('liDB,aDlJ->aBiJ',LTliDA,T2ab)
    tmpab += 0.5*einsum('lJdB,adil->aBiJ',LTlIdA,T2aa)
    tmpab += 0.5*einsum('LJDB,aDiL->aBiJ',LTLIDA,T2ab)

    Pabij += tmpaa - tmpaa.transpose((0,1,3,2)) - tmpaa.transpose((1,0,2,3)) + tmpaa.transpose((1,0,3,2))
    PABIJ += tmpbb - tmpbb.transpose((0,1,3,2)) - tmpbb.transpose((1,0,2,3)) + tmpbb.transpose((1,0,3,2))
    PaBiJ += tmpab

    T2tempaa = T2aa + einsum('cj,ai->acij',T1a,T1a) - einsum('ci,aj->acij',T1a,T1a)
    T2tempab = T2ab + einsum('cj,ai->acij',T1b,T1a)
    T2tempbb = T2bb + einsum('cj,ai->acij',T1b,T1b) - einsum('ci,aj->acij',T1b,T1b)
    Lcb = einsum('klcd,bdkl->cb',L2aa,T2aa)
    Lcb += 2.0*einsum('kLcD,bDkL->cb',L2ab,T2ab)
    LCB = einsum('klcd,bdkl->cb',L2bb,T2bb)
    LCB += 2.0*einsum('lKdC,dBlK->CB',L2ab,T2ab)
    tmpaa = -0.5*einsum('cb,acij->abij',Lcb,T2tempaa)
    tmpbb = -0.5*einsum('cb,acij->abij',LCB,T2tempbb)
    tmpab = -0.5*einsum('CB,aCiJ->aBiJ',LCB,T2tempab)
    tmpab -= 0.5*einsum('ca,cBiJ->aBiJ',Lcb,T2tempab)
    Pabij += tmpaa - tmpaa.transpose((1,0,2,3))
    PABIJ += tmpbb - tmpbb.transpose((1,0,2,3))
    PaBiJ += tmpab

    Lkj = einsum('klcd,cdjl->kj',L2aa,T2aa)
    Lkj += 2.0*einsum('kLcD,cDjL->kj',L2ab,T2ab)
    LKJ = einsum('klcd,cdjl->kj',L2bb,T2bb)
    LKJ += 2.0*einsum('lKdC,dClJ->KJ',L2ab,T2ab)
    tmpaa = -0.5*einsum('kj,abik->abij',Lkj,T2tempaa)
    tmpbb = -0.5*einsum('kj,abik->abij',LKJ,T2tempbb)
    tmpab = -0.5*einsum('KJ,aBiK->aBiJ',LKJ,T2tempab)
    tmpab -= 0.5*einsum('ki,aBkJ->aBiJ',Lkj,T2tempab)
    Pabij += tmpaa - tmpaa.transpose((0,1,3,2))
    PABIJ += tmpbb - tmpbb.transpose((0,1,3,2))
    PaBiJ += tmpab

    T2tempaa = T2aa + einsum('ci,dj->cdij',T1a,T1a)
    T2tempab = T2ab + einsum('ci,dj->cdij',T1a,T1b)
    T2tempbb = T2bb + einsum('ci,dj->cdij',T1b,T1b)
    LToo = einsum('klcd,cdij->klij',L2aa,T2tempaa)
    LTOO = einsum('klcd,cdij->klij',L2bb,T2tempbb)
    LToO = einsum('kLcD,cDiJ->kLiJ',L2ab,T2tempab)
    LToO += einsum('kLdC,dCiJ->kLiJ',L2ab,T2ab)
    LTOo = -einsum('lKdC,dCiJ->KliJ',L2ab,T2ab)
    LTOo -= einsum('lKcD,cDiJ->KliJ',L2ab,T2tempab)
    tmpaa = einsum('klij,ak->alij',LToo,T1a)
    tmpaa = 0.25*einsum('alij,bl->abij',tmpaa,T1a)
    tmpbb = einsum('klij,ak->alij',LTOO,T1b)
    tmpbb = 0.25*einsum('alij,bl->abij',tmpbb,T1b)
    tmpab1 = einsum('kLiJ,ak->aLiJ',LToO,T1a)
    tmpab2 = einsum('KliJ,AK->AliJ',LTOo,T1b)
    tmpab = 0.25*einsum('aLiJ,BL->aBiJ',tmpab1,T1b)
    tmpab -= 0.25*einsum('BliJ,al->aBiJ',tmpab2,T1a)
    Pabij += tmpaa - tmpaa.transpose((1,0,2,3))
    PABIJ += tmpbb - tmpbb.transpose((1,0,2,3))
    PaBiJ += tmpab

    Looov = einsum('klcd,ci->klid',L2aa,T1a)
    LOOOV = einsum('klcd,ci->klid',L2bb,T1b)
    LoOoV = einsum('kLcD,ci->kLiD',L2ab,T1a)
    LoOOv = -einsum('kLdC,CI->kLId',L2ab,T1b)
    Loooo = einsum('klid,dj->klij',Looov,T1a)
    Loooo -= Loooo.transpose((0,1,3,2))
    LOOOO = einsum('klid,dj->klij',LOOOV,T1b)
    LOOOO -= LOOOO.transpose((0,1,3,2))
    LoOoO = einsum('kLiD,DJ->kLiJ',LoOoV,T1b)
    LoOoO -= einsum('kLJd,di->kLiJ',LoOOv,T1a)
    Pabij += 0.25*einsum('klij,abkl->abij',Loooo,T2tempaa)
    PABIJ += 0.25*einsum('klij,abkl->abij',LOOOO,T2tempbb)
    PaBiJ += 0.25*einsum('kLiJ,aBkL->aBiJ',LoOoO,T2tempab)
    PaBiJ += 0.25*einsum('lKiJ,aBlK->aBiJ',LoOoO,T2ab)

    Lalid = einsum('klcd,ak,ci->alid',L2aa,T1a,T1a)
    LALID = einsum('klcd,ak,ci->alid',L2bb,T1b,T1b)
    LaLiD = einsum('kLcD,ak,ci->aLiD',L2ab,T1a,T1a)
    LAlId = einsum('lKdC,AK,CI->AlId',L2ab,T1b,T1b)
    LaLId = -einsum('kLdC,ak,CI->aLId',L2ab,T1a,T1b)
    LAliD = -einsum('lKcD,AK,ci->AliD',L2ab,T1b,T1a)
    tmpaa = einsum('alid,bdjl->abij',Lalid,T2aa)
    tmpaa += einsum('aLiD,bDjL->abij',LaLiD,T2ab)
    tmpbb = einsum('alid,bdjl->abij',LALID,T2bb)
    tmpbb += einsum('AlId,dBlJ->ABIJ',LAlId,T2ab)

    tmpab = einsum('alid,dBlJ->aBiJ',Lalid,T2ab)
    tmpab += einsum('aLiD,BDJL->aBiJ',LaLiD,T2bb)
    tmpab += einsum('aLJd,dBiL->aBiJ',LaLId,T2ab)
    tmpab += einsum('BliD,aDlJ->aBiJ',LAliD,T2ab)
    tmpab += einsum('BlJd,adil->aBiJ',LAlId,T2aa)
    tmpab += einsum('BLJD,aDiL->aBiJ',LALID,T2ab)

    Pabij -= tmpaa
    Pabij += tmpaa.transpose((0,1,3,2)) + tmpaa.transpose((1,0,2,3))
    Pabij -= tmpaa.transpose((1,0,3,2))
    PABIJ -= tmpbb
    PABIJ += tmpbb.transpose((0,1,3,2)) + tmpbb.transpose((1,0,2,3))
    PABIJ -= tmpbb.transpose((1,0,3,2))
    PaBiJ -= tmpab
    return Pabij,PABIJ,PaBiJ

def ccsd_pt_simple(F,I,eo,ev,T1,T2):
    raise Exception("ccsd(T) is not implemented")
    no = eo.shape[0]
    nv = ev.shape[0]
    d =(no*nv)**3
    T3 = numpy.einsum('adij,bcdk->abcijk',T2,I.vvvo)
    T3 -= T3.transpose((0,1,2,3,5,4))
    T3 -= T3.transpose((0,1,2,5,4,3))
    T3 -= T3.transpose((1,0,2,3,4,5))
    T3 -= T3.transpose((2,1,0,3,4,5))

    T3 += numpy.einsum('abij,cljk->abcijk',T2,I.vooo)
    D = 1/(eo[None,None,None,:,None,None] + eo[None,None,None,None,:,None] + eo[None,None,None,None,None,:]
            - ev[:,None,None,None,None,None] - ev[None,:,None,None,None,None]- ev[None,None,:,None,None,None])

    Et = (1.0/36.0)*numpy.einsum('abcijk,abcijk,abcijk->',T3,D,T3) \
            + 0.25*numpy.einsum('ai,bcjk,abcijk->',T1,I.vvoo,T3)
    return Et

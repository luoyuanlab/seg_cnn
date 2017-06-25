""" yluo - 06/04/2016 creation
stats and evaluation utilities. 
"""
__author__= """Yuan Luo (yuan.hypnos.luo@gmail.com)"""
__revision__="0.5"

import numpy as np

def confMat(gt, mc, nc):
    ## assumes gt and mc are in vector format
    ## columns: predicted as, rows: actual
    gt = np.asarray(gt); mc = np.asarray(mc)
    ni = len(gt)
    lcgt = np.unique(gt)
    nimc = len(mc)
    lcmc = np.unique(mc)
    # print('ni %s, nc %s, nimc %s, ncmc %s' % (ni, nc, nimc, ncmc))
    assert (nimc == ni) 
    cm = np.zeros((nc, nc)) ## gt as rows
    for i in lcgt:
        for j in lcmc:
            cm[i,j] = np.sum((gt==i) * (mc==j))
    return (cm)


def cmPRF(cm, ncstart=0):
    # calculate precision, recall and f-measure given result output
    # ncstart controls whether to include 
    nc, nc2 = cm.shape
    assert nc==nc2
    pres = np.zeros(nc); recs = np.zeros(nc); f1s = np.zeros(nc)

    tp_a = 0; fn_a = 0; fp_a = 0
    for c in range(ncstart,nc):
        tp = cm[c,c]; tp_a += tp
        mask = np.ones(nc,dtype=bool)
        mask[c] = 0
        fn = np.sum( cm[c, mask] ); fn_a += fn
        fp = np.sum( cm[mask, c] ); fp_a += fp
        if tp+fp == 0:
            pre = 1
        else:
            pre = tp / (tp+fp)
        if tp+fn == 0:
            rec = 1
        else:
            rec = tp / (tp+fn)
        if pre+rec == 0:
            f = 0
        else:
            f = 2*pre*rec / (pre+rec)
        pres[c] = pre; recs[c] = rec; f1s[c] = f
    if tp_a+fp_a == 0:
        mipre = 1
    else:
        mipre = tp_a / (tp_a+fp_a)
    if tp_a+fn_a == 0:
        mirec = 1
    else:
        mirec = tp_a / (tp_a+fn_a)
    if mipre+mirec == 0:
        mif = 0
    else:
        mif = 2*mipre*mirec / (mipre+mirec)
    return (pres, recs, f1s, mipre, mirec, mif)

def rsig(seq, probe):
    R = len(seq)
    r = sum(seq >= probe)
    print(r)
    p = (r+1.)/(R+1.)
    ## p = r/R
    return p;

def randPairedSigTest(out1, out2, N=1000):
    row = len(out1)
    mean1 = np.mean(out1); mean2 = np.mean(out2)
    pmean1 = [0] * N; pmean2 = [0] * N

    perm1 = out1;
    perm2 = out2;
    
    for i in range(N):
        coins = np.random.rand(row); ## only pick one half 
        ind1 = 0.5 <= coins;
        ind2 = 0.5 > coins;
        
        perm1[ind1] = out1[ind1];
        perm1[ind2] = out2[ind2];
        
        perm2[ind1] = out2[ind1];
        perm2[ind2] = out1[ind2];
        
        pmean1[i] = mean(perm1)
        pmean2[i] = mean(perm2)
    
    ## p = rsig( abs(pmean1 - pmean2), abs(mean1 - mean2) );
    p = rsig( pmean1 - pmean2, mean1 - mean2 ); # one side test
    print('rand paired sig test on mean p=%f\n' % (p))
    return p;






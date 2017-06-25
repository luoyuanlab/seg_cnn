""" yluo - 05/01/2016 creation
Preprocess i2b2/VA relations to generate data files ready to used by Seg-CNN
"""
__author__= """Yuan Luo (yuan.hypnos.luo@gmail.com)"""
__revision__="0.5"

import numpy as np
import cPickle
from collections import defaultdict
import sys, re, os
import pandas as pd
import data_util as du


trp_rel = ['TrIP', 'TrWP', 'TrCP', 'TrAP', 'TrNAP']
tep_rel = ['TeRP', 'TeCP']
pp_rel = ['PIP']
dosages = ['mg', 'bid', 'prn', 'qd', 'po', 'tid', 'qhs', 'qid', 'qod']

def load_stoplist(fn):
    ## in case you want to use stopword list, not really needed
    h = {}
    f = open(fn)
    for ln in f:
        swd = ln.rstrip(' \n')
        if not h.has_key(swd):
            h[swd] = 0
        h[swd] += 1
    return h;

def include_wd(wd):
    ans = re.search(r'[A-Za-z]', wd) != None or wd in '/:;()[]{}-+?'
    ans &= not 'year-old' in wd and 'y/o' != wd and 'yo' != wd and 'y.o.' != wd
    ans &= not '&' in wd and not '**' in wd
    ans &= re.search(r'[0-9]', wd) == None 
    ans &= re.search(r'^[^A-Za-z].+', wd) == None # cannot start with nonletter
    return ans;

def clean_wds(wdsin, hstop={}, strict=True):
    wdsout = []
    for wd in wdsin:
        if not strict or include_wd(wd):
            wd = re.sub('"', '', wd)
            wd = du.removeNonAscii(wd)
            if not hstop.has_key(wd):
                if strict and ('-' in wd and '-' != wd):
                    for swd in wd.split('-'):
                        if len(swd)>1:
                            wdsout.append(swd)
                else: 
                    if strict and len(wd)>1:
                        wd = re.sub('[^A-Za-z]*$', '', wd)
                    if len(wd)>0:
                        wdsout.append(wd)
    return wdsout;

def load_mask(words, iid):
    hmask = {}
    wid = 0
    st = -1; end = -1
    for word in words:
        if re.search(r'^\*\*.*\[', word):
            st = wid
        if st != -1 and re.search(r'\]', word): # can start and end on the same word
            end = wid
            hmask[(st, end+1)] = 1
            if st == -1:
                print('mask err in %s' % (iid))
            st = -1
        wid += 1
    if st != -1:
        print('mask unfinished in %s' % (iid))
        hmask[(st, wid+1)] = 1
    return hmask;


def inmask(i, hmask):
    res = False
    for mask in hmask:
        mst = mask[0]
        mend = mask[1]
        if mst <= i and i < mend:
            res = True
            break
    return res;

def mask_concept(words, st, end, hmask, hproblem, htreatment, htest):
    mwds = [words[i] for i in range(st, end) if not inmask(i, hproblem) and not inmask(i, htreatment) and not inmask(i, htest) and not inmask(i, hmask)]
    return mwds;

def fmask(words, st, end, hmask={}, hproblem={}, htreatment={}, htest={}, mask=False, skip_concept=False):
    mwds = []; out = True
    for i in range(st, end):
        if not (mask and inmask(i, hmask)):
            if skip_concept and inmask(i, hproblem):
                if out:
                    mwds += ['problem']
                out = False
            elif skip_concept and inmask(i, htreatment):
                if out:
                    mwds += ['treatment']
                out = False
            elif skip_concept and inmask(i, htest):
                if out:
                    mwds += ['test']
                out = False
            else:
                mwds += [words[i]]
                out = True
    return mwds;

def concept_type(cs, ce, iid, hproblem, htreatment, htest):
    if (cs, ce) in hproblem:
        ct = 'problem'
    elif (cs, ce) in htreatment:
        ct = 'treatment'
    elif (cs, ce) in htest:
        ct = 'test'
    else:
        print('should not be here %s %s in %s' % (cs, ce, iid))
    return ct;

def markup_sen(words, c1s, c1e, c2s, c2e, iid, hproblem, htreatment, htest):
    c1t = concept_type(c1s, c1e, iid, hproblem, htreatment, htest)
    c2t = concept_type(c2s, c2e, iid, hproblem, htreatment, htest)
    if c1s < c2s:
        mwords = words[:c1s] + ['['] + words[c1s:c1e] + [']%s' % (c1t)] + words[c1e:c2s] + ['['] + words[c2s:c2e] + [']%s' % (c2t)] + words[c2e:]
    else:
        mwords = words[:c2s] + ['['] + words[c2s:c2e] + [']%s' % (c2t)] + words[c2e:c1s] + ['['] + words[c1s:c1e] + [']%s' % (c1t)] + words[c1e:]
    return mwords;
        
def build_inst(iid, c1s, c1e, c2s, c2e, sen, vocab, hlen, rel='None', padlen=0, hstop={}, hproblem={}, htreatment={}, htest={}, mask=False, skip_concept=False):
    words = sen.split()
    hmask = load_mask(words, iid)
    c1 = clean_wds(fmask(words, c1s, c1e, hmask), hstop, strict=False)
    c2 = clean_wds(fmask(words, c2s, c2e, hmask), hstop, strict=False)
    c1t = concept_type(c1s, c1e, iid, hproblem, htreatment, htest)
    c2t = concept_type(c2s, c2e, iid, hproblem, htreatment, htest)
    cts = '_'.join(sorted((c1t, c2t)))
    prec_end = min(c1s, c2s); succ_start = max(c1e,c2e)
    prec = clean_wds(fmask(words, 0, prec_end, hmask, hproblem, htreatment, htest, mask, skip_concept), hstop) 
    succ = clean_wds(fmask(words, succ_start, len(words), hmask, hproblem, htreatment, htest, mask, skip_concept), hstop) 
    mid = clean_wds(fmask(words, min(c1e,c2e), max(c1s,c2s), hmask, hproblem, htreatment, htest, mask, skip_concept), hstop)
    hlen[cts]['c1'] = max(hlen[cts]['c1'], len(c1))
    hlen[cts]['c2'] = max(hlen[cts]['c2'], len(c2))
    hlen[cts]['mid'] = max(hlen[cts]['mid'], len(mid))

    if c1s < c2s:
        c1 = prec[-padlen:] + c1 + mid[:padlen]
        c2 = mid[-padlen:] + c2 + succ[:padlen]
    else:
        c1 = mid[-padlen:] + c1 + succ[:padlen]
        c2 = prec[-padlen:] + c2 + mid[:padlen]

    prec = prec[-padlen:]
    succ = succ[:padlen]
    hlen[cts]['prec'] = max(hlen[cts]['prec'], len(prec))
    hlen[cts]['succ'] = max(hlen[cts]['succ'], len(succ))

    mwords = markup_sen(words, c1s, c1e, c2s, c2e, iid, hproblem, htreatment, htest)
    datum  = {'iid':iid,
              'rel':rel, 
              'c1': c1,
              'c2': c2,
              'prec': prec,
              'succ': succ,
              'mid': mid,
              'sen': ' '.join(mwords)}
    return datum;

def add_none_rel(fn, hpair, sens, rels, vocab, hlen, mask=False, mid_lmax=None, padlen=0, hstop={}, hproblem={}, htreatment={}, htest={}, skip_concept=False):
    for senid in hpair:
        for con_pair in hpair[senid]:
            c1s = con_pair[0][0]
            c1e = con_pair[0][1]
            c2s = con_pair[1][0]
            c2e = con_pair[1][1]
            sen = sens[senid].lower()
            iid = '%s:%s (%d,%d) (%d,%d)' % (fn, senid, c1s, c1e, c2s, c2e)
            midlen = max(c1s,c2s) - min(c1e,c2e)
            if mid_lmax != None and midlen > mid_lmax:
                continue

            datum = build_inst(iid, c1s, c1e, c2s, c2e, sen, vocab, hlen, padlen=padlen, hstop=hstop, hproblem=hproblem[senid], htreatment=htreatment[senid], htest=htest[senid], mask=mask, skip_concept=skip_concept)

            if datum != None:
                rels.append(datum)

            
def load_con(fncon, htrp, htep, hpp):
    hproblem = defaultdict(list)
    htreatment = defaultdict(list)
    htest = defaultdict(list)
    lc = 0
    with open(fncon, 'r') as f:
        lc += 1
        for ln in f:
            ln = ln.rstrip(' \n')
            mo = re.search(r'c=".*?" (\d+):(\d+) \d+:(\d+)\|\|t="(.*?)"', ln)
            if mo:
                senid = int(mo.group(1))-1 # start with 1 in annotation
                cs = int(mo.group(2))
                ce = int(mo.group(3))+1
                ctype = mo.group(4)
                if ctype == "problem":
                    hproblem[senid].append((cs, ce))
                elif ctype == "treatment":
                    htreatment[senid].append((cs, ce))
                elif ctype == "test":
                    htest[senid].append((cs,ce))
                else:
                    print('unrecognized ctype %s at %d in %s' % (ctype, lc, fncon))

        # sort the concepts according to positions
        for senid in htreatment:
            htreatment[senid] = sorted(htreatment[senid], key=lambda x: x[0])
        for senid in htest:
            htest[senid] = sorted(htest[senid], key=lambda x: x[0])
        for senid in hproblem:
            hproblem[senid] = sorted(hproblem[senid], key=lambda x: x[0])

        # genereate all possible pairs for treatment problem
        for senid in htreatment:
            for tr in htreatment[senid]:
                for p in hproblem[senid]:
                    htrp[senid][(tr, p)] = 1
                    
        # test problem pair
        for senid in htest:
            for te in htest[senid]:
                for p in hproblem[senid]:
                    htep[senid][(te, p)] = 1
                    
        # problem pair
        for senid in hproblem:
            for i in range(len(hproblem[senid])-1):
                for j in range(i+1,len(hproblem[senid])):
                    p1 = hproblem[senid][i]
                    p2 = hproblem[senid][j]
                    if p1[0] < p2[0]:
                        hpp[senid][(p1, p2)] = 1
                    else:
                        print('collapsed %s and %s in %d in %s' % (p1, p2, senid, fncon))
    return (hproblem, htreatment, htest)

def load_rel(fnrel, sens, htrp, htep, hpp, vocab, hlen, trp_data, tep_data, pp_data, mask=False, padlen=0, hstop={}, hproblem={}, htreatment={}, htest={}, skip_concept=False, pip_reorder=False):
    sen_seen = {}
    fnroot = re.sub(r'^.*/', '', fnrel)
    with open(fnrel, 'r') as f:
        lc = 0; trp_mid_lmax = 0; tep_mid_lmax = 0; pp_mid_lmax = 0
        for ln in f:
            ln = ln.rstrip(' \n')
            lc += 1
            mo = re.search(r'c=".*?" (\d+):(\d+) \d+:(\d+)\|\|r="(.*?)"\|\|c=".*?" \d+:(\d+) \d+:(\d+)', ln)
            if mo:
                senid = int(mo.group(1))-1 # start with 1 in annotation
                
                c1s = int(mo.group(2))
                c1e = int(mo.group(3))+1
                rel = mo.group(4)
                c2s = int(mo.group(5))
                c2e = int(mo.group(6))+1
                sen = sens[senid].lower()
                if not sen_seen.has_key(senid):
                    words = sen.split()
                    for word in set(clean_wds(words, hstop)):
                        vocab[word] += 1
                    sen_seen[senid] = 1
                iid = '%s:%s (%d,%d) (%d,%d)' % (fnroot, senid, c1s, c1e, c2s, c2e)
                if pip_reorder and rel == 'PIP':
                    datum = build_inst(iid, min(c1s,c2s), min(c1e,c2e), max(c1s,c2s), max(c1e,c2e), sen, vocab, hlen, rel, padlen=padlen, hstop=hstop, hproblem=hproblem[senid], htreatment=htreatment[senid], htest=htest[senid], mask=mask, skip_concept=skip_concept)
                else:
                    datum = build_inst(iid, c1s, c1e, c2s, c2e, sen, vocab, hlen, rel, padlen=padlen, hstop=hstop, hproblem=hproblem[senid], htreatment=htreatment[senid], htest=htest[senid], mask=mask, skip_concept=skip_concept)
                midlen = max(c1s,c2s) - min(c1e,c2e)
                con_pair = ((c1s, c1e), (c2s, c2e))
                con_pair2 = ((c2s, c2e), (c1s, c1e))
                if rel in trp_rel:
                    trp_data.append(datum)
                    if not htrp[senid].has_key(con_pair):
                        print('no trp pair %s in %s' % (con_pair,iid))
                    htrp[senid].pop(con_pair, None)
                    trp_mid_lmax = max(trp_mid_lmax, midlen)
                elif rel in tep_rel:
                    tep_data.append(datum)
                    if not htep[senid].has_key(con_pair):
                        print('no tep pair %s in %s' % (con_pair,iid))
                    htep[senid].pop(con_pair, None)
                    tep_mid_lmax = max(tep_mid_lmax, midlen)
                elif rel in pp_rel:
                    pp_data.append(datum)
                    if not hpp[senid].has_key(con_pair) and not hpp[senid].has_key(con_pair2):
                        print('no pp pair %s in %s' % (con_pair,iid))
                    hpp[senid].pop(con_pair, None)
                    hpp[senid].pop(con_pair2, None)
                    pp_mid_lmax = max(pp_mid_lmax, midlen)
                else:
                    print('unrecognized rel %s' % (rel))
            else:
                print('non-matching line %d in %s' % (lc, fnrel))
        add_none_rel(fnroot, htrp, sens, trp_data, vocab, hlen, mask=mask, padlen=padlen, hstop=hstop, hproblem=hproblem, htreatment=htreatment, htest=htest, skip_concept=skip_concept) 
        add_none_rel(fnroot, htep, sens, tep_data, vocab, hlen, mask=mask, padlen=padlen, hstop=hstop, hproblem=hproblem, htreatment=htreatment, htest=htest, skip_concept=skip_concept) 
        add_none_rel(fnroot, hpp, sens, pp_data, vocab, hlen, mask=mask, padlen=padlen, hstop=hstop, hproblem=hproblem, htreatment=htreatment, htest=htest, skip_concept=skip_concept) 
    return;

                            
def build_data(dn, vocab, hlen, mask=False, padlen=0, hstop={}, skip_concept=False, pip_reorder=False):
    """
    Loads data 
    """
    trp_data = [] # problem treatment
    tep_data = [] # problem test
    pp_data = [] # problem problem
    dntxt = '%s/txt' % (dn)
    dnrel = '%s/rel' % (dn)
    dncon = '%s/concept' % (dn)
    fc = 0
    
    for fntxt in os.listdir(dntxt):
        htrp = defaultdict(dict)
        htep = defaultdict(dict)
        hpp = defaultdict(dict)
        
        if not(re.search(r'.txt$', fntxt)):
            continue
        fc += 1
        fnrel = re.sub('.txt', '.rel', fntxt)
        fncon = re.sub('.txt', '.con', fntxt)
        sens = []
        with open('%s/%s' % (dntxt, fntxt), "r") as f:
            for ln in f:
                ln = ln.rstrip(' \n')
                sens.append(ln)

        (hproblem, htreatment, htest) = load_con('%s/%s' % (dncon, fncon), htrp, htep, hpp)
                            
        load_rel('%s/%s' % (dnrel, fnrel), sens, htrp, htep, hpp, vocab, hlen, trp_data, tep_data, pp_data, mask=mask, padlen=padlen, hstop=hstop, hproblem=hproblem, htreatment=htreatment, htest=htest, skip_concept=skip_concept, pip_reorder=pip_reorder)

    print(fc)
    return trp_data, tep_data, pp_data

def build_train_test(cdn='/n/data1/hms/dbmi/zaklab/yluo/semrel', hlen = defaultdict(lambda:defaultdict(float)), padlen=0, fnstop=None, skip_concept=False, pip_reorder=False):
    hstop = {}; vocab = defaultdict(float)
    if fnstop != None:
        hstop=load_stoplist(fnstop)
    trp_upmcp_tr, tep_upmcp_tr, pp_upmcp_tr = build_data('%s/Relation_Challenge_Data/Released/concept_assertion_relation_training/upmcp' % (cdn), vocab, hlen, mask=True, padlen=padlen, hstop=hstop, skip_concept=skip_concept, pip_reorder=pip_reorder)
    print('upmcp_tr %d' % (len(trp_upmcp_tr)))
    trp_upmcd_tr, tep_upmcd_tr, pp_upmcd_tr = build_data('%s/Relation_Challenge_Data/Released/concept_assertion_relation_training/upmcd' % (cdn), vocab, hlen, mask=True, padlen=padlen, hstop=hstop, skip_concept=skip_concept, pip_reorder=pip_reorder)
    print('upmcd_tr %d' % (len(trp_upmcd_tr)))
    trp_beth_tr, tep_beth_tr, pp_beth_tr = build_data('%s/Relation_Challenge_Data/Released/concept_assertion_relation_training/beth' % (cdn), vocab, hlen, mask=True, padlen=padlen, hstop=hstop, skip_concept=skip_concept, pip_reorder=pip_reorder)
    print('beth_tr %d' % (len(trp_beth_tr)))
    trp_partners_tr, tep_partners_tr, pp_partners_tr = build_data('%s/Relation_Challenge_Data/Released/concept_assertion_relation_training/partners' % (cdn), vocab, hlen, padlen=padlen, hstop=hstop, skip_concept=skip_concept, pip_reorder=pip_reorder)
    print('partners_tr %d' % (len(trp_partners_tr)))
    trp_rel_te, tep_rel_te, pp_rel_te = build_data('%s/Relation_Challenge_Data/Released/concept_assertion_relation_test' % (cdn), vocab, hlen, mask=True, padlen=padlen, hstop=hstop, skip_concept=skip_concept, pip_reorder=pip_reorder)
    trp_rel_tr = trp_upmcp_tr + trp_upmcd_tr + trp_beth_tr + trp_partners_tr
    tep_rel_tr = tep_upmcp_tr + tep_upmcd_tr + tep_beth_tr + tep_partners_tr
    pp_rel_tr = pp_upmcp_tr + pp_upmcd_tr + pp_beth_tr + pp_partners_tr
    
    return trp_rel_tr, tep_rel_tr, pp_rel_tr, trp_rel_te, tep_rel_te, pp_rel_te, vocab, hlen


def embed_train_test(fnem, fnwid='../data/vocab.txt', fndata='../data/semrel.p', padlen=0, fnstop=None, skip_concept=True, pip_reorder=False, binEmb=False):
    trp_rel_tr, tep_rel_tr, pp_rel_tr, trp_rel_te, tep_rel_te, pp_rel_te, vocab, hlen = build_train_test(padlen=padlen, fnstop=fnstop, skip_concept=skip_concept, pip_reorder=pip_reorder)
    fwid = open(fnwid, 'w')
    for wd in sorted(vocab.keys()):
        if vocab[wd] >= 1:
            fwid.write('%s\n' % (wd))
        else:
            vocab.pop(wd, None)
    fwid.close()
    if binEmb:
        mem, hwoov, hwid = du.load_bin_vec(fnem, fnwid)
    else:
        mem, hwoov, hwid = du.indexEmbedding(fnem, fnwid)
    mem = mem.astype('float32')
    # the saved data are lists of relation dicts, with keys c1, c2 ,etc.
    cPickle.dump([trp_rel_tr, tep_rel_tr, pp_rel_tr, trp_rel_te, tep_rel_te, pp_rel_te, vocab, dict(hlen), mem, hwoov, hwid], open(fndata, "wb"))
    print "dataset created!"
    return mem, hwoov, hwid;



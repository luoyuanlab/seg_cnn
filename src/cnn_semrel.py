""" 
This file was extensively rewritten from the sentence CNN code at https://github.com/yoonkim/CNN_sentence by Yoon Kim
"""
__author__= """Yuan Luo (yuan.hypnos.luo@gmail.com)"""
__revision__="0.5"

import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
import re
import warnings
import sys
import time
import os
import stats_util as su
warnings.filterwarnings("ignore")   

htrp_rel = {'TrIP':1, 'TrWP':2, 'TrCP':3, 'TrAP':4, 'TrNAP':5, 'None':0}
htep_rel = {'TeRP':1, 'TeCP':2, 'None':0}
hpp_rel = {'PIP':1, 'None':0}


#different non-linearities
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
def Tanh(x):
    y = T.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)

def make_rel_hash(drel):
    hrel = {}
    for rel in drel:
        hrel[rel['iid']] = rel
    return hrel;
       
def train_conv_net(datasets, rel_tr, rel_te, hlen,
                   U, # yluo: embedding matrix
                   fnres,
                   img_w=300, 
                   filter_hs=[3,4,5],
                   hidden_units=[100,2], # hidden_units[1] is number of classes
                   dropout_rate=[0.5],
                   shuffle_batch=True,
                   n_epochs=25, 
                   batch_size=50, # yluo: how many sentences to extract to compute gradient
                   lr_decay = 0.95,
                   conv_non_linear="relu",
                   activations=[Iden],
                   sqr_norm_lim=9,
                   non_static=True):
    """
    Train a simple conv net
    img_h = sentence length (padded where necessary)
    img_w = word vector length (300 for word2vec)
    filter_hs = filter window sizes    
    hidden_units = [x,y] x is the number of feature maps (per filter window), and y is the penultimate layer
    sqr_norm_lim = s^2 in the paper
    lr_decay = adadelta decay parameter
    """
    hrel_tr = make_rel_hash(rel_tr)
    hrel_te = make_rel_hash(rel_te)
    rng = np.random.RandomState()
    img_h_tot = len(datasets[0][0])-2
    pad = max(filter_hs) - 1
    filter_w = img_w
    # yluo: what does different feature maps correspond to?
    feature_maps = hidden_units[0]
    filter_shapes = []

    for filter_h in filter_hs:
        # yluo: what does 1 in the filter shape mean?
        # (number of filters, num input feature maps, filter height, filter width)
        # how to interpet different filters?
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))

        parameters = [("image shape",img_h_tot,img_w),
                      ("filter shape",filter_shapes),
                      ("hidden_units",hidden_units),
                      ("dropout", dropout_rate), ("batch_size",batch_size),
                      ("non_static", non_static), ("learn_decay",lr_decay),
                      ("conv_non_linear", conv_non_linear),
                      ("non_static", non_static),
                      ("sqr_norm_lim",sqr_norm_lim),
                      ("shuffle_batch",shuffle_batch)]
    print parameters    
    
    #define model architecture
    index = T.lscalar()
    # x = T.matrix('x')
    c1 = T.matrix('c1')
    c2 = T.matrix('c2')
    prec = T.matrix('prec')
    mid = T.matrix('mid')
    succ = T.matrix('succ')
    y = T.ivector('y')
    iid = T.vector('iid')
    Words = theano.shared(value = U, name = "Words")
    zero_vec_tensor = T.vector()
    zero_vec = np.zeros(img_w)
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0,:], zero_vec_tensor))], allow_input_downcast=True)
    c1_input = Words[T.cast(c1.flatten(),dtype="int32")].reshape((c1.shape[0],1,c1.shape[1],Words.shape[1])) # reshape to 3d array
    c2_input = Words[T.cast(c2.flatten(),dtype="int32")].reshape((c2.shape[0],1,c2.shape[1],Words.shape[1])) # reshape to 3d array
    prec_input = Words[T.cast(prec.flatten(),dtype="int32")].reshape((prec.shape[0],1,prec.shape[1],Words.shape[1])) # reshape to 3d array
    mid_input = Words[T.cast(mid.flatten(),dtype="int32")].reshape((mid.shape[0],1,mid.shape[1],Words.shape[1])) # reshape to 3d array
    succ_input = Words[T.cast(succ.flatten(),dtype="int32")].reshape((succ.shape[0],1,succ.shape[1],Words.shape[1])) # reshape to 3d array
    layer0_input = {'c1':c1_input, 'c2':c2_input, 'prec':prec_input, 'mid':mid_input, 'succ':succ_input}                                  
    conv_layers = []
    layer1_inputs = []
    
    for i in xrange(len(filter_hs)):
        for seg in hlen.keys(): # used hlen as a global var, to fix 
            filter_shape = filter_shapes[i]
            img_h = hlen[seg]+2*pad
            pool_size = (img_h-filter_h+1, img_w-filter_w+1)
            conv_layer = LeNetConvPoolLayer(rng, input=layer0_input[seg],
                                            image_shape=(batch_size, 1, img_h, img_w),
                                            filter_shape=filter_shape,
                                            poolsize=pool_size,
                                            non_linear=conv_non_linear)
            layer1_input = conv_layer.output.flatten(2) # yluo: 2 dimensions
            conv_layers.append(conv_layer) # yluo: layer 0
            layer1_inputs.append(layer1_input) # yluo: 3 dimensions
    layer1_input = T.concatenate(layer1_inputs,1) # yluo: 2 dimensions
    hidden_units[0] = feature_maps*len(filter_hs)*len(hlen)    
    classifier = MLPDropout(rng, input=layer1_input, layer_sizes=hidden_units, activations=activations, dropout_rates=dropout_rate)
    
    #define parameters of the model and update functions using adadelta
    params = classifier.params     
    for conv_layer in conv_layers:
        params += conv_layer.params
    if non_static:
        #if word vectors are allowed to change, add them as model parameters
        params += [Words]
    cost = classifier.negative_log_likelihood(y) 
    dropout_cost = classifier.dropout_negative_log_likelihood(y)           
    grad_updates = sgd_updates_adadelta(params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)
    
    #shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate , stochastic gradient descent
    #extra data (at random)
    
    tr_size = datasets[0].shape[0]
    hi_seg = datasets[2]
    print(hi_seg)
    c1s, c1e = hi_seg['c1']; c2s, c2e = hi_seg['c2']; mids, mide = hi_seg['mid']
    precs, prece = hi_seg['prec']; succs, succe = hi_seg['succ']
    yi = hi_seg['y']; idi = hi_seg['iid']
    if tr_size % batch_size > 0:
        extra_data_num = batch_size - tr_size % batch_size
        train_set = rng.permutation(datasets[0])   
        extra_data = train_set[:extra_data_num]
        new_data=np.append(datasets[0],extra_data,axis=0)
    else:
        new_data = datasets[0]
    new_data = rng.permutation(new_data)
    n_batches = new_data.shape[0]/batch_size
    n_train_batches = int(np.round(n_batches*0.9))
    
    #divide train set into train/val sets
    c1_te = datasets[1][:, c1s:c1e]
    c2_te = datasets[1][:, c2s:c2e]
    prec_te = datasets[1][:, precs:prece]
    mid_te = datasets[1][:, mids:mide]
    succ_te = datasets[1][:, succs:succe]
    test_set = datasets[1]
    y_te = np.asarray(test_set[:,yi],"int32")
    
    train_set = new_data[:n_train_batches*batch_size,:]
    val_set = new_data[n_train_batches*batch_size:,:]     
    x_tr, y_tr = shared_dataset((train_set[:,:img_h_tot], train_set[:,-1]))
    x_val, y_val = shared_dataset((val_set[:,:img_h_tot], val_set[:,-1]))
    iid_tr = train_set[:,idi].flatten()
    iid_val = val_set[:,idi].flatten()
    iid_te = test_set[:,idi].flatten()
    print('len iid_val %d' % (len(iid_val)))

    n_val_batches = n_batches - n_train_batches
    
    #compile theano functions to get train/val/test errors    
    val_model = theano.function([index], classifier.preds(y),
        givens={
            c1: x_val[index*batch_size: (index+1)*batch_size, c1s:c1e],
            c2: x_val[index*batch_size: (index+1)*batch_size, c2s:c2e],
            prec: x_val[index*batch_size: (index+1)*batch_size, precs:prece],
            mid: x_val[index*batch_size: (index+1)*batch_size, mids:mide],
            succ: x_val[index*batch_size: (index+1)*batch_size, succs:succe],
            y: y_val[index*batch_size: (index+1)*batch_size],
        },
        allow_input_downcast=True, on_unused_input='warn')
    # this test_model is batch test model for train
    test_model = theano.function([index], classifier.errors(y),
        givens={
            c1: x_tr[index*batch_size: (index+1)*batch_size, c1s:c1e],
            c2: x_tr[index*batch_size: (index+1)*batch_size, c2s:c2e],
            prec: x_tr[index*batch_size: (index+1)*batch_size, precs:prece],
            mid: x_tr[index*batch_size: (index+1)*batch_size, mids:mide],
            succ: x_tr[index*batch_size: (index+1)*batch_size, succs:succe],
            y: y_tr[index*batch_size: (index+1)*batch_size]},
                                 allow_input_downcast=True)               
    train_model = theano.function([index], cost, updates=grad_updates,
        givens={
            c1: x_tr[index*batch_size: (index+1)*batch_size, c1s:c1e],
            c2: x_tr[index*batch_size: (index+1)*batch_size, c2s:c2e],
            prec: x_tr[index*batch_size: (index+1)*batch_size, precs:prece],
            mid: x_tr[index*batch_size: (index+1)*batch_size, mids:mide],
            succ: x_tr[index*batch_size: (index+1)*batch_size, succs:succe],
            y: y_tr[index*batch_size: (index+1)*batch_size]},
        allow_input_downcast = True)     
    test_pred_layers = []
    test_size = len(y_te)
    c1_te_input = Words[T.cast(c1.flatten(),dtype="int32")].reshape((c1_te.shape[0],1,c1_te.shape[1],Words.shape[1]))
    c2_te_input = Words[T.cast(c2.flatten(),dtype="int32")].reshape((c2_te.shape[0],1,c2_te.shape[1],Words.shape[1]))
    prec_te_input = Words[T.cast(prec.flatten(),dtype="int32")].reshape((prec_te.shape[0],1,prec_te.shape[1],Words.shape[1]))
    mid_te_input = Words[T.cast(mid.flatten(),dtype="int32")].reshape((mid_te.shape[0],1,mid_te.shape[1],Words.shape[1]))
    succ_te_input = Words[T.cast(succ.flatten(),dtype="int32")].reshape((succ_te.shape[0],1,succ_te.shape[1],Words.shape[1]))
    test_layer0_input = {'c1':c1_te_input, 'c2':c2_te_input, 'prec':prec_te_input, 'mid':mid_te_input, 'succ':succ_te_input}

    cl_id = 0 # conv layer id
    for i in xrange(len(filter_hs)):
        for seg in hlen.keys():
            conv_layer = conv_layers[cl_id]
            test_layer0_output = conv_layer.predict(test_layer0_input[seg], test_size) ## doesn't seeem to matter if just use layer0_input here
            test_pred_layers.append(test_layer0_output.flatten(2))
            cl_id += 1
    test_layer1_input = T.concatenate(test_pred_layers, 1)
    test_y_pred = classifier.predict(test_layer1_input)
    test_error = T.mean(T.neq(test_y_pred, y))
    test_model_all = theano.function([c1,c2,prec,mid,succ], test_y_pred, allow_input_downcast = True)  
    
    #start training over mini-batches
    print '... training'
    epoch = 0
    best_val_perf = 0
    test_perf = 0       
    cost_epoch = 0    
    while (epoch < n_epochs):
        start_time = time.time()
        epoch = epoch + 1
        if shuffle_batch:
            for minibatch_index in rng.permutation(range(n_train_batches)):
                cost_epoch = train_model(minibatch_index)
                set_zero(zero_vec)
        else:
            for minibatch_index in xrange(n_train_batches):
                cost_epoch = train_model(minibatch_index)  
                set_zero(zero_vec)
        train_losses = [np.mean(test_model(i)) for i in xrange(n_train_batches)]
        train_perf = 1 - np.mean(train_losses)
        val_preds = np.asarray([])
        for i in xrange(n_val_batches):
            val_sb_preds = val_model(i)
            y_sb = y_val[i*batch_size:(i+1)*batch_size].eval()
            val_sb_errors = val_sb_preds != y_sb
            err_ind = [j for j,x in enumerate(val_sb_errors) if x==1]
            val_sb = iid_val[i*batch_size:(i+1)*batch_size]
            val_preds = np.append(val_preds, val_sb_preds)

        val_perf = 1- np.mean(y_val.eval() != val_preds)
        val_cm = su.confMat(y_val.eval(), val_preds, hidden_units[1])

        (val_pres, val_recs, val_f1s, val_mipre, val_mirec, val_mif) = su.cmPRF(val_cm, ncstart=1)
        print('epoch: %i, training time: %.2f secs, train perf: %.2f %%, mif: %.2f %%' % (epoch, time.time()-start_time, train_perf * 100., val_mif*100.)) 
        if val_mif >= best_val_perf:
            best_val_perf = val_mif
            test_pred = test_model_all(c1_te,c2_te,prec_te,mid_te,succ_te)
            test_errors = test_pred != y_te
            err_ind = [j for j,x in enumerate(test_errors) if x==1]
            test_cm = su.confMat(y_te, test_pred, hidden_units[1])
            print('\n'.join([''.join(['{:10}'.format(int(item)) for item in row]) 
                             for row in test_cm]))
            (pres, recs, f1s, mipre, mirec, mif) = su.cmPRF(test_cm, ncstart=1)
            print('mipre %s, mirec %s, mif %s' % (mipre, mirec, mif))
    cPickle.dump([y_te,test_pred], open(fnres, "wb"))
    return (mipre, mirec, mif)

def shared_dataset(data_xy, iid=None, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                            dtype=theano.config.floatX),
                                borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                            dtype=theano.config.floatX),
                                borrow=borrow)
        if iid == None:
            return shared_x, T.cast(shared_y, 'int32')
        else:
            shared_iid = theano.shared(np.asarray(iid), borrow=borrow)
            return shared_x, T.cast(shared_y, 'int32'), shared_iid
        
def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9,word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param      
    return updates 

def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)
    
def safe_update(dict_to, dict_from):
    """
    re-make update dictionary for safe updating
    """
    for key, val in dict(dict_from).iteritems():
        if key in dict_to:
            raise KeyError(key)
        dict_to[key] = val
    return dict_to
    
def get_idx_from_segment(words, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x

def merge_segs(c1, c2, prec, mid, succ, y, iid, over_sampling=False, down_sampling=None):
    rng = np.random.RandomState()
    hi_seg = {}
    cursor = 0
    print('shapes c1: %s, c2: %s, prec: %s, mid: %s, succ: %s, iid: %s, y: %s' % (c1.shape, c2.shape, prec.shape, mid.shape, succ.shape, iid.shape, y.shape))
    data = np.hstack((c1, c2, prec, mid, succ, iid, y))
    hi_seg['c1'] = [cursor,c1.shape[1]]; cursor += c1.shape[1]
    hi_seg['c2'] = [cursor, cursor+c2.shape[1]]; cursor += c2.shape[1]
    hi_seg['prec'] = [cursor, cursor+prec.shape[1]]; cursor += prec.shape[1]
    hi_seg['mid'] = [cursor, cursor+mid.shape[1]]; cursor += mid.shape[1]
    hi_seg['succ'] = [cursor, cursor+succ.shape[1]]; cursor += succ.shape[1]
    hi_seg['iid'] = cursor; cursor += 1
    hi_seg['y'] = cursor
    y = y.flatten()
    if over_sampling:
        num_none = np.sum(y==0)
        data_os = data
        for c in np.unique(y):
            if c != 0:
                num_c = np.sum(y==c)
                num_sample = num_none - num_c
                print(data.shape)
                data_c = data[np.asarray(y==c),:]
                print('data_c lab %s' % (data_c[:,hi_seg['y']].flatten()[1:20]))
                while num_sample > num_c:
                    data_os = np.vstack((data_os, data_c))
                    num_sample -= num_c
                data_os = np.vstack((data_os, data_c[:num_sample,:]))
        data = data_os
        print('over-sampled dist %s %s' % (np.unique(data[:,hi_seg['y']], return_counts=True)))
    if down_sampling != None:
        data_ds = None
        (labs, counts) = np.unique(y, return_counts=True)
        cnt_min = min(counts)
        for (lab, count) in zip(labs, counts):
            data_c = data[np.asarray(y==lab),:]
            if count > down_sampling*cnt_min:
                data_c = data_c[rng.permutation(count)[:down_sampling*cnt_min],:]
            if data_ds == None:
                data_ds = data_c
            else:
                data_ds = np.vstack((data_ds, data_c))
        data = data_ds
        print('down-sampled dist %s %s' % (np.unique(data[:,hi_seg['y']], return_counts=True)))
    return data, hi_seg;

def make_idx_data_train_test(rel_tr, rel_te, word_idx_map, hlen, hrel, k=300, filter_h=5, down_sampling=None):
    """
    Transforms sentences into a 2-d matrix.
    """
    c1_tr, c1_te = [], []
    c2_tr, c2_te = [], []
    prec_tr, prec_te = [], []
    mid_tr, mid_te = [], []
    succ_tr, succ_te = [], []
    y_tr, y_te = [], []
    iid_tr, iid_te = [], []
    for rel in rel_tr:
        c1_tr.append( get_idx_from_segment(rel['c1'], word_idx_map, hlen['c1'], k, filter_h) )
        c2_tr.append( get_idx_from_segment(rel['c2'], word_idx_map, hlen['c2'], k, filter_h) )
        prec_tr.append( get_idx_from_segment(rel['prec'], word_idx_map, hlen['prec'], k, filter_h) )
        mid_tr.append( get_idx_from_segment(rel['mid'], word_idx_map, hlen['mid'], k, filter_h) )
        succ_tr.append( get_idx_from_segment(rel['succ'], word_idx_map, hlen['succ'], k, filter_h) )
        y_tr.append(hrel[rel['rel']])
        iid_tr.append(rel['iid'])
    print(np.unique(y_tr, return_counts=True))
    y_tr = np.asarray(y_tr); y_tr = y_tr.reshape(len(y_tr), 1)
    iid_tr = np.asarray(iid_tr); iid_tr = iid_tr.reshape(len(iid_tr), 1)
    
    c1_tr_lens = map(len, c1_tr)
    print('c1 tr len max %d, min %d' % (max(c1_tr_lens), min(c1_tr_lens)))
    for rel in rel_te:
        c1_te.append( get_idx_from_segment(rel['c1'], word_idx_map, hlen['c1'], k, filter_h) )
        c2_te.append( get_idx_from_segment(rel['c2'], word_idx_map, hlen['c2'], k, filter_h) )
        prec_te.append( get_idx_from_segment(rel['prec'], word_idx_map, hlen['prec'], k, filter_h) )
        mid_te.append( get_idx_from_segment(rel['mid'], word_idx_map, hlen['mid'], k, filter_h) )
        succ_te.append( get_idx_from_segment(rel['succ'], word_idx_map, hlen['succ'], k, filter_h) )
        y_te.append(hrel[rel['rel']])
        iid_te.append(rel['iid'])
    print(np.unique(y_te, return_counts=True))
    y_te = np.asarray(y_te); y_te = y_te.reshape(len(y_te), 1)
    iid_te = np.asarray(iid_te); iid_te = iid_te.reshape(len(iid_te), 1)
    
    c1_tr = np.array(c1_tr,dtype="int"); c1_te = np.array(c1_te,dtype="int")
    c2_tr = np.array(c2_tr,dtype="int"); c2_te = np.array(c2_te,dtype="int")
    prec_tr = np.array(prec_tr,dtype="int"); prec_te = np.array(prec_te,dtype="int")
    mid_tr = np.array(mid_tr,dtype="int"); mid_te = np.array(mid_te,dtype="int")
    succ_tr = np.array(succ_tr,dtype="int"); succ_te = np.array(succ_te,dtype="int")
    train, hi_seg_tr = merge_segs(c1_tr, c2_tr, prec_tr, mid_tr, succ_tr, y_tr, iid_tr, down_sampling=down_sampling)
    test, hi_seg_te = merge_segs(c1_te, c2_te, prec_te, mid_te, succ_te, y_te, iid_te)
    return [train, test, hi_seg_tr, hi_seg_te]     
  
   
if __name__=="__main__":
    img_w = sys.argv[3]
    mo = re.search('-img_w(\d+)', img_w)
    if mo:
        img_w = int(mo.group(1))
    else:
        print('example: -img_w300')
        sys.exit(1)

    l1_nhu = sys.argv[4]
    mo = re.search('-l1_nhu(\d+)', l1_nhu)
    if mo:
        l1_nhu = int(mo.group(1)) # number of hidden units first layer
    else:
        print('example: -l1_nhu100')
        sys.exit(1)

    pad = sys.argv[5]
    mo = re.search('-pad(\d+)', pad)
    if mo:
        pad = int(mo.group(1))
    else:
        print('example: -pad5')
        sys.exit(1)
        
    task = sys.argv[6]
    
    fndata = '../data/semrel_pp/semrel_pp%s_pad%s.p' % (img_w, pad)
    fdata = open(fndata,"rb")
    x = cPickle.load(fdata)
    fdata.close()
    trp_rel_tr, tep_rel_tr, pp_rel_tr, trp_rel_te, tep_rel_te, pp_rel_te, vocab, hlen, mem, hwoov, hwid = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10]
    for cts in hlen.keys():
        hlen[cts]['c1'] += 2*pad
        hlen[cts]['c2'] += 2*pad
    print('msg: %s loaded!' % (fndata))
    mode= sys.argv[1]
    word_vectors = sys.argv[2]    
    if mode=="-nonstatic":
        print "model architecture: CNN-non-static"
        non_static=True
    elif mode=="-static":
        print "model architecture: CNN-static"
        non_static=False
    execfile("cnn_classes.py")    
    if word_vectors=="-word2vec":
        print "using: word2vec vectors"
        U = mem
    else:
        print "unrecognized word_vectors option: %s" % (word_vectors)
        
    results = []

    if task=='-trp':
        trp_data = make_idx_data_train_test(trp_rel_tr, trp_rel_te, hwid, hlen['problem_treatment'], htrp_rel, k=img_w, filter_h=5, down_sampling=None)
        (mipre, mirec, mif) = train_conv_net(trp_data, trp_rel_tr, trp_rel_te,
                                             hlen['problem_treatment'],
                                             U,
                                             fnres='../result/trp_img%s_nhu%s_pad%s.p' % (img_w, l1_nhu, pad),
                                             img_w=img_w,
                                             lr_decay=0.95,
                                             filter_hs=[3,4,5], 
                                             conv_non_linear="relu",
                                             hidden_units=[l1_nhu,6], 
                                             shuffle_batch=True, 
                                             n_epochs=35, 
                                             sqr_norm_lim=9,
                                             non_static=non_static,
                                             batch_size=50,
                                             dropout_rate=[0.5])
        print("msg: trp img_w: %s, l1_nhu: %s, pad: %s, mipre: %s, mirec: %s, mif: %s" % (img_w, l1_nhu, pad, mipre, mirec, mif))

    if task=='-tep':
        tep_data = make_idx_data_train_test(tep_rel_tr, tep_rel_te, hwid, hlen['problem_test'], htep_rel, k=img_w, filter_h=5)
        (mipre, mirec, mif) = train_conv_net(tep_data,tep_rel_tr, tep_rel_te,
                                             hlen['problem_test'],
                                             U,
                                             fnres='../result/tep_img%s_nhu%s_pad%s.p' % (img_w, l1_nhu, pad),
                                             img_w=img_w,
                                             lr_decay=0.95,
                                             filter_hs=[3,4,5],
                                             conv_non_linear="relu",
                                             hidden_units=[l1_nhu,3], 
                                             shuffle_batch=True, 
                                             n_epochs=25, 
                                             sqr_norm_lim=9,
                                             non_static=non_static,
                                             batch_size=50,
                                             dropout_rate=[0.5])
        print("msg: tep img_w: %s, l1_nhu: %s, pad: %s, mipre: %s, mirec: %s, mif: %s" % (img_w, l1_nhu, pad, mipre, mirec, mif))

    if task=='-pp':
        pp_data = make_idx_data_train_test(pp_rel_tr, pp_rel_te, hwid, hlen['problem_problem'], hpp_rel, k=img_w, filter_h=5, down_sampling=4) 
        (mipre, mirec, mif) = train_conv_net(pp_data, pp_rel_tr, pp_rel_te,
                                             hlen['problem_problem'],
                                             U,
                                             fnres='../result/pp_img%s_nhu%s_pad%s.p' % (img_w, l1_nhu, pad),
                                             img_w=img_w,
                                             lr_decay=0.95,
                                             filter_hs=[3,4,5],
                                             conv_non_linear="relu",
                                             hidden_units=[l1_nhu,2], 
                                             shuffle_batch=True, 
                                             n_epochs=35, 
                                             sqr_norm_lim=9,
                                             non_static=non_static,
                                             batch_size=50,
                                             dropout_rate=[0.5])
        print("msg: pp img_w: %s, l1_nhu: %s, pad: %s, mipre: %s, mirec: %s, mif: %s" % (img_w, l1_nhu, pad, mipre, mirec, mif))

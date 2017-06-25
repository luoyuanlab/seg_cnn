""" yluo - 05/01/2016 creation
Call cnn_preprocess functions to generate data files ready to used by Seg-CNN
"""
__author__= """Yuan Luo (yuan.hypnos.luo@gmail.com)"""
__revision__="0.5"

import cnn_preprocess as cp
import re
img_w = 200
pad = 7
fnwem = '../data/embedding/mimic3_pp%s.txt' % (img_w) 
fndata='../data/semrel_pp/semrel_pp%s_pad%s.p' % (img_w, pad)

mem, hwoov, hwid = cp.embed_train_test(fnwem, fndata=fndata, padlen=pad)


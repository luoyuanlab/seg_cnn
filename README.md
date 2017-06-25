# Segment CNN for relation classification
### Requirements
Code is written in Python (2.7) and requires Theano (0.9).


### Data Preprocessing
To process the raw data, run
```
python data.py 
```

This is a wrapper code calling `cnn_preprocess.embed_train_test()` with arguments specifying word embedding dimensions (e.g., 200) and padding length (e.g., 7). 
This will create a pickle object (e.g., `semrel_pp200_pad7.p`) in the directory 'data/semrel_pp', which contains the dataset
with the right components to be used by `cnn_semrel.py`.



### Using the GPU
GPU will result in a good 10x to 20x speed-up compared to CPU, so it is highly recommended. 
For example:
```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,gcc.cxxflags=-march=corei7 python cnn_semrel.py -static -word2vec -img_w200 -l1_nhu100 -pad7 -trp
```


### Example output
GPU output (mif is micro-averaged f-measure):
```
epoch: 1, training time: 4.86 secs, train perf: 70.54 %, mif: 55.50 %
epoch: 2, training time: 4.70 secs, train perf: 76.67 %, mif: 61.89 %
epoch: 3, training time: 4.75 secs, train perf: 78.77 %, mif: 64.75 %
```



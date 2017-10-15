import lstm
import bucket_io
import mxnet as mx
import numpy as np
import logging
from rnn_model import LSTMInferenceModel
# Each line contains at most 129 chars. 
seq_len=129
# embedding dimension, which maps a character to a 256-dimension vector
num_embed=256
# number of lstm layers
num_lstm_layer=3
# hidden unit in LSTM cell
num_hidden=512
 
symbol= lstm.lstm_unroll(
    num_lstm_layer, 
    seq_len,
    len(vocab)+1,
    num_hidden=num_hidden,
    num_embed=num_embed,
    num_label=len(vocab)+1,
    dropout=0.2)

batch_size=32
 
# initalize states for LSTM
init_c= [('l%d_init_c'%l, (batch_size, num_hidden)) for l inrange(num_lstm_layer)]
init_h= [('l%d_init_h'%l, (batch_size, num_hidden)) for l inrange(num_lstm_layer)]
init_states= init_c + init_h
 
# Even though BucketSentenceIter supports various length examples,
# we simply use the fixed length version here
data_train= bucket_io.BucketSentenceIter(
    "./obama.txt",
    vocab, 
    [seq_len], 
    batch_size,             
    init_states, 
    seperate_char='\n',
    text2id=text2id,
    read_content=read_content)

logging.getLogger().setLevel(logging.DEBUG)
 
# We will show a quick demo with only 1 epoch. In practice, we can set it to be 100
num_epoch=1
# learning rate 
learning_rate=0.01
 
# Evaluation metric
def Perplexity(label, pred):
    loss =0.
    for i inrange(pred.shape[0]):
        loss +=-np.log(max(1e-10, pred[i][int(label[i])]))
    return np.exp(loss/ label.size)
 
model= mx.model.FeedForward(
    ctx=mx.gpu(0),
    symbol=symbol,
    num_epoch=num_epoch,
    learning_rate=learning_rate,
    momentum=0,
    wd=0.0001,
    initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))
 
model.fit(X=data_train,
          eval_metric=mx.metric.np(Perplexity),
          batch_end_callback=mx.callback.Speedometer(batch_size,20),
          epoch_end_callback=mx.callback.do_checkpoint("obama"))

# helper strcuture for prediction
def MakeRevertVocab(vocab):
    dic = {}
    for k, v in vocab.items():
        dic[v] = k
    return dic
 
# make input from char
def MakeInput(char, vocab, arr):
    idx = vocab[char]
    tmp = np.zeros((1,))
    tmp[0]= idx
    arr[:] = tmp
 
# helper function for random sample 
def _cdf(weights):
    total =sum(weights)
    result = []
    cumsum =0
    for w in weights:
        cumsum += w
        result.append(cumsum/ total)
    return result
 
def _choice(population, weights):
    assertlen(population)==len(weights)
    cdf_vals = _cdf(weights)
    x = random.random()
    idx = bisect.bisect(cdf_vals, x)
    return population[idx]
 
# we can use random output or fixed output by choosing largest probability
def MakeOutput(prob, vocab, sample=False, temperature=1.):
    if sample ==False:
        idx = np.argmax(prob, axis=1)[0]
    else:
        fix_dict = [""]+ [vocab[i] for i inrange(1,len(vocab)+1)]
        scale_prob = np.clip(prob,1e-6,1-1e-6)
        rescale = np.exp(np.log(scale_prob)/ temperature)
        rescale[:] /= rescale.sum()
        return _choice(fix_dict, rescale[0, :])
    try:
        char = vocab[idx]
    except:
        char =''
    return char
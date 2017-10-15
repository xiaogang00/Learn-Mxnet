import os
import urllib
import zipfile

if not os.path.exists("char_lstm.zip"):
    urllib.urlretrieve("http://data.mxnet.io/data/char_lstm.zip","char_lstm.zip")

with zipfile.ZipFile("char_lstm.zip","r")as f:
    f.extractall("./")    

with open('obama.txt','r') as f:
    print f.read()[0:1000]

def read_content(path):
    with open(path)as ins:        
        return ins.read()

# Return a dict which maps each char into an unique int id
def build_vocab(path):
    content =list(read_content(path))
    idx =1# 0 is left for zero-padding
    the_vocab = {}
    for word in content:
        if len(word)==0:
            continue
        if not word in the_vocab:
            the_vocab[word] = idx
            idx +=1
    return the_vocab

# Encode a sentence with int ids
def text2id(sentence, the_vocab):
    words =list(sentence)
    return [the_vocab[w] for w in words if len(w)>0]

# build char vocabluary from input
vocab= build_vocab("./obama.txt")
print('vocab size = %d'%(len(vocab)))
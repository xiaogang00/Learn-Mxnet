import rnn_model
 
# load from check-point
_, arg_params, __ = mx.model.load_checkpoint("obama",75)
 
# build an inference model
model= rnn_model.LSTMInferenceModel(
    num_lstm_layer,
    len(vocab)+1,
    num_hidden=num_hidden,
    num_embed=num_embed,
    num_label=len(vocab)+1,
    arg_params=arg_params,
    ctx=mx.gpu(),
    dropout=0.2)

seq_length=600
input_ndarray= mx.nd.zeros((1,))
revert_vocab= MakeRevertVocab(vocab)
# Feel free to change the starter sentence
output='The United States'
random_sample=False
new_sentence=True
 
ignore_length=len(output)

for i inrange(seq_length):
    if i <= ignore_length -1:
        MakeInput(output[i], vocab, input_ndarray)
    else:
        MakeInput(output[-1], vocab, input_ndarray)
    prob = model.forward(input_ndarray, new_sentence)
    new_sentence =False
    next_char = MakeOutput(prob, revert_vocab, random_sample)
    if next_char =='':
        new_sentence =True
    if i >= ignore_length -1:
        output += next_char
print(output)
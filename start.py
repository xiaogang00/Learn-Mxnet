import mxnet as mx 

#声明训练和校验数据集的数据迭代器  
train = mx.io.MNISTIter(
    image = "mnist/train-images-idx3-ubyte",
    label = "mnist/train-labels-idx1-ubyte",
    batch_size = 128,
    data_shape = (784, )
)

# 声明两层的MLP
# 在这里的堆叠也很像是keras中的函数式贯序模型
data = mx.symbol.Variable('data')
fc1  = mx.symbol.FullyConnected(data= data, num_hidden=128)
act1 = mx.symbol.Activation(data= fc1, act_type="relu")
fc2  = mx.symbol.FullyConnected(data= act1, num_hidden =64)
act2 = mx.symbol.Activation(data= fc2, act_type="relu")
fc3  = mx.symbol.FullyConnected(data= act2, num_hidden=10)
mlp  = mx.symbol.SoftmaxOutput(data= fc3, name ='softmax')

#训练模型
model = mx.model.FeedForward(
    symbol = mlp,
    num_epoch = 20,
    learning_rate = .1
)

model.fit(X = train, eval_data = val)

#进行预测
test = mx.io.MNISTIter(
    mage = "mnist/test-images-idx3-ubyte",
    label = "mnist/test-labels-idx1-ubyte",
    batch_size = 128,
    data_shape = (784, )
)
model.predict(X = test)

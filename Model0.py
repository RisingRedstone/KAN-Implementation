# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:17:53 2024

@author: prath
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:15:43 2024

@author: prath
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from Network import KAN8Layer as KANLayer
from Network import KANParameters as KANParam
import tensorflow_datasets as tfds

class Model(tf.Module):
    def __init__(self, ParameterLayers, seed = None, name = ""):
        super().__init__(name=name)
        self.KANLayers = []
        i = 0
        self.seed = seed
        self.ParameterLayers = ParameterLayers
        self.trimData = [(tf.ones([ParameterLayers[0].inpDim], tf.int32), 
                          tf.ones([ParameterLayers[0].inpDim], tf.int32))]
        for param in ParameterLayers:
            self.KANLayers.append(KANLayer(param.inpDim, 
                                           param.outDim, 
                                           G = param.G, 
                                           p=param.p, 
                                           intGridRange = param.intGridRange,
                                           intGrid = param.intGrid,
                                           seed = seed, 
                                           name = ""+"_Layer"+str(i)))
            self.trimData.append((tf.ones(param.outDim, tf.int32), tf.ones(param.outDim, tf.int32)))
            i+=1
            
    def pruneNeurons(self, epsilon = 1e-2):
        self.trimData[0] = tf.ones(tf.shape(self.trimData[0][1]), tf.int32)
        for i in range(1, len(self.trimData)-1):
            self.trimData[i] = tf.where(self.trimData[i][0] >= epsilon, 1, 0) & tf.where(self.trimData[i][1] >= epsilon, 1, 0)
        self.trimData[-1] = tf.ones(tf.shape(self.trimData[-1][0]), tf.int32)
        for i in range(len(self.trimData)-1):
            inpMask = self.trimData[i]
            outMask = self.trimData[i+1]
            inpSiz = tf.cast(tf.reduce_sum(self.trimData[i]), tf.int32).numpy()
            outSiz = tf.cast(tf.reduce_sum(self.trimData[i+1]), tf.int32).numpy()
            Vars = {"Phi": self.KANLayers[i].Phi, "W": self.KANLayers[i].W}
            Vars["Phi"] = tf.boolean_mask(Vars["Phi"], inpMask, axis = 0)
            Vars["W"] = tf.boolean_mask(Vars["W"], inpMask, axis = 0)
            Vars["Phi"] = tf.boolean_mask(Vars["Phi"], outMask, axis = 1)
            Vars["W"] = tf.boolean_mask(Vars["W"], outMask, axis = 1)
            self.ParameterLayers[i].inpDim = inpSiz
            self.ParameterLayers[i].outDim = outSiz
            self.KANLayers[i] = KANLayer(self.ParameterLayers[i].inpDim, 
                                         self.ParameterLayers[i].outDim, 
                                         G = self.ParameterLayers[i].G, 
                                         p = self.ParameterLayers[i].p, 
                                         intGrid = self.ParameterLayers[i].intGrid,
                                         intGridRange = self.ParameterLayers[i].intGridRange,
                                         seed = self.seed, 
                                         name = ""+"_Layer"+str(i))
            self.KANLayers[i].assignValues(Vars)
    
    def gridChange(self, index, gridRange, G, p, intGridRange, stddev = 0.01):
        Vars = self.KANLayers[index].UpdateGrid(G, p, gridRange, stddev = stddev)
        self.ParameterLayers[index].G = G
        self.ParameterLayers[index].p = p
        self.ParameterLayers[index].gridRange = gridRange
        self.ParameterLayers[index].intGridRange = intGridRange
        param = self.ParameterLayers[index]
        self.KANLayers[index] = KANLayer(param.inpDim, 
                                         param.outDim,
                                         G = param.G,
                                         p = param.p,
                                         gridRange = param.gridRange,
                                         intGridRange=param.intGridRange,
                                         intGrid = param.intGrid,
                                         seed = self.seed,
                                         name = ""+"_Layer"+str(index))
        self.KANLayers[index].assignValues(Vars)
    
    def __call__(self, X):
        X = tf.keras.layers.Flatten()(X)
        regloss = tf.constant(0.0)
        i = 0
        self.PointCount = []
        #self.trimData[0] = tf.ones([self.ParameterLayers[0][0]], tf.int32)
        for kan in self.KANLayers:
            X, temploss, Neur, pointCount = kan(X)
            self.PointCount.append(pointCount)
            regloss = regloss + temploss
            self.trimData[i] = (self.trimData[i][0], Neur[0])
            self.trimData[i+1] = (Neur[1], Neur[1])
            i+=1
        return (tf.keras.activations.softmax(X, axis = 1), regloss)
        #return X

'''        
Params = [[784, 30, 2, 1],
         [30, 20, 30, 2],
         [20, 10, 50, 3]]
'''                      

#Params = [[784, 64, 5, 3, 100],
#          [64, 10, 5, 3, 100]]

Params = [KANParam(784, 64, G=5, p=3, intGrid=30, intGridRange=[-2.0, 2.0]),
          KANParam(64, 10, G=5, p=3, intGrid=100, intGridRange=[-10.0, 10.0])]

((train_X, train_Y), (test_X, test_Y)) = tf.keras.datasets.mnist.load_data(
    path='mnist.npz'
)

train_epoch_X = []
train_epoch_Y = []
N = 10000
i = 0
while((i+N) <= 60000):
    a = train_Y[i:(i+N)]
    b = np.zeros((a.size, a.max() + 1))
    b[np.arange(a.size), a] = 1
    train_epoch_X.append( (train_X[i:(i+N)]/255) )
    train_epoch_Y.append(b)
    i+=N



def test_acc(mod, X, Y):
    pred_Y, L = mod(X)
    Y_pred_vals = tf.argmax(pred_Y, axis = 1)
    Acc = tf.math.reduce_mean(tf.where(tf.math.equal(Y, Y_pred_vals), 1.0, 0.0)) * 100
    print("Test Accuracy: ", Acc.numpy(),"%")
    return Acc.numpy()
    

model = Model(Params, seed = 100, name = "Mod1")
#loss = tf.keras.losses.MSE
loss = tf.keras.losses.CategoricalCrossentropy()
initial_learning_rate = 0.01
weight_regularization_loss = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=6,
    decay_rate=0.99,
    staircase=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

AccScores = []
TestAccScores = []
i = 0
epochs = 10
NonCount = 1
for _ in range(epochs):
    for X, Y in zip(train_epoch_X, train_epoch_Y):
        print("epoch: ",i, end = "\t")
        with tf.GradientTape() as tape:
            Y_pred, regloss = model(X)
            mse_loss = loss(Y, Y_pred) + weight_regularization_loss * regloss

        
        Y_pred_vals = tf.argmax(Y_pred, axis = 1)
        Y_true_vals = tf.argmax(Y, axis = 1)
        Acc = tf.math.reduce_mean(tf.where(tf.math.equal(Y_true_vals, Y_pred_vals), 1.0, 0.0)) * 100
        AccScores.append(Acc.numpy())
        print("Acc: ", Acc.numpy(),"%", end = "\t")
        print(mse_loss.numpy(), weight_regularization_loss * regloss.numpy(), end = "\t")
        
        print("LR: ", optimizer.lr.numpy(), end = "\t")
        dY_dM = tape.gradient(mse_loss, model.trainable_variables)


        #Correct Nan Values and set them to 0
        grads = []
        NonCOunt = 1
        for grad in dY_dM:
            grads.append(tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad))
            if(tf.reduce_any(tf.math.is_nan(grad) == True)):
                print("NAN_Exists ", NonCOunt)
                NonCOunt = -10000
            NonCOunt += 1
        print()
        if(NonCOunt < 0):
            break
                
        optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))
        i+=1
    
    if(NonCOunt < 0):
        break
    TestAccScores.append(test_acc(model, tf.cast(test_X, tf.float32)/255, test_Y))

plt.plot([x for x in range(len(AccScores))], AccScores)
plt.plot([(len(train_epoch_X) * (x+1)) for x in range(len(TestAccScores))], TestAccScores)


C = [(mod[0].numpy(), mod[1].numpy()) for mod in model.trimData]
B = [lis.numpy() for lis in model.PointCount]
#C[0] = np.reshape(C[0], (28, 28))
def Plot0(I, b = 5):
    fig = plt.figure(figsize = (8, 8))
    CDat= (np.reshape(C[0][1], (28, 28)) - tf.reduce_min(C[0][1])) / (tf.reduce_max(C[0][1]) - tf.reduce_min(C[0][1]))
    for i in range(1, b+1):
        fig.add_subplot(b, 2, (2*i)-1)
        plt.imshow(train_X[I+i])
        fig.add_subplot(b, 2, 2*i)
        plt.imshow(train_X[I+i]*CDat)
    plt.plot()

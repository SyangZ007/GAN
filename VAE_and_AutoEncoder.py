import numpy as np # linear algebra
import tensorflow as tf

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import Model

#datasets
(x_train, _), (x_test, _) = fashion_mnist.load_data()
#归一化
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
print(x_train.shape,x_test.shape)

train_set=tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(256)
test_set=tf.data.Dataset.from_tensor_slices(x_test).batch(256)
#VAE 变分自动编码器
#VAE有两种生成图像方式：重构:model(x) 和 取样sample:model.decoder(z)
z_dims = 64 
class VAE(Model):
    def __init__(self):
        super(VAE,self).__init__()
        #encoder
        self.fc1 = layers.Dense(128,activation='relu')
        self.fc2 = layers.Dense(z_dims)#线性激活  输出mean
        self.fc3 = layers.Dense(z_dims)#线性激活  输出var
        #decoder
        self.fc4 = layers.Dense(128,activation='relu')
        self.out = layers.Dense(28*28)#线性激活输出
        
    def encoder(self,x):
        #encoder have 2 output: mean&variance
        h = self.fc1(x)
        mean = self.fc2(h)#get mean
        log_var = self.fc3(h)#get log var,经过log变化，值域为(-∞,∞)
        return mean,log_var
    
    def decoder(self,z):
        #decoder 输入为mean ,var采样得到的z
        out = self.fc4(z)
        return self.out(out)
    
    def reparameterize(self,mean,log_var):
        #随机采样获取的mean,var反向传播过程不可导，本函数作为trick，近似随机采样
        eps = tf.random.normal(tf.shape(log_var))#正态分布的eps
        std = tf.math.exp(log_var)**0.5 #获取标准差
        z = mean + std*eps
        return z
    
    def call(self,x,training=None):
        #encoder
        mean,log_var = self.encoder(x)#encoder have 2 output: mean&variance
        #reparameterize get z
        z = self.reparameterize(mean,log_var)
        #decoder
        x_hat = self.decoder(z)
        #返回重构的x_hat,(mean,var)用于模型学习
        return x_hat,mean,log_var

#training
from IPython import display
model=VAE()
#model.build(input_shape=(None,28*28))
opt=tf.keras.optimizers.Adam()

def train(epochs):
    '''training with visualization'''
    losses=[]
    for epoch in range(epochs):
        for step,x in enumerate(train_set):
            #input:x shape:batch,28,28>>>batch,784;mean;var
            with tf.GradientTape() as tape:
                x=tf.reshape(x,shape=(-1,784))
                x_hat,mean,log_var=model(x)
                #计算loss 
                #x_hat未经sigmoid激活
                rec_loss=tf.keras.losses.binary_crossentropy(x,x_hat,from_logits=True)#重构loss
                rec_loss=tf.reduce_mean(rec_loss)
                #comput KL Divergence(mean,var) - N(0,1),使模型输出的分布学习N(0,1)分布
                #KL散度 q(x)>(mean,var)能在多大程度上表达p(x)>N(0,1)所包含的信息，KL散度越大，表达效果越差
                kl_div=-0.5*(log_var+1-mean**2-tf.math.exp(log_var))#简化后的公式
                kl_div=tf.reduce_mean(kl_div)
                
                loss=rec_loss + 1.*kl_div
            grads=tape.gradient(loss,model.trainable_variables)
            opt.apply_gradients(zip(grads,model.trainable_variables))
            if step%100 ==0:
                print('epoch:{},step:{},loss:{}'.format(epoch,step,loss))
                losses.append(loss)
        #每个epoch结束检验一次模型重构能力以及根据随机z向量生成图像能力
        #检验模型重构能力
        display.clear_output(wait=True)
        for x_rec in test_set.take(1):
            x_rec=tf.reshape(x_rec,(-1,784))
            out,_,_=model(x_rec)
            out=tf.sigmoid(out)*255.
            out=tf.reshape(out,(-1,28,28))

            plt.figure(figsize=(6,6))
            for i in range(16):
                plt.subplot(4,4,i+1)
                plt.imshow(out[i])
            plt.title('reconstruct image',loc='center',y=4.6)
            plt.show()
        print('\n')
        #指定z向量，模型decoder生成一张随机图片（不是重构）
        z=tf.random.normal((16,64))
        out=model.decoder(z)
        out=tf.sigmoid(out)*255.
        out=tf.reshape(out,(-1,28,28))

        plt.figure(figsize=(6,6))
        for i in range(16):
            plt.subplot(4,4,i+1)
            plt.imshow(out[i])
        plt.title('generate image with random vector Z',loc='center',y=4.6)
        plt.show()
    return losses

losses=train(epochs=10)

plt.plot(losses)
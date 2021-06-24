import os
import numpy as np 
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model,layers

##############################dataset################
imgs_path=glob('../input/animefacedataset/images/*.jpg')
print(len(imgs_path))#63565
batch_size=64

@tf.function
def preprocess(path):
    img=tf.io.read_file(path)
    img=tf.image.decode_image(img,channels=3,dtype='float32',expand_animations=False)
    img=tf.image.resize(img,size=[64,64])
    img=tf.clip_by_value(img,0,1)
    #数据集图像已经做了归一化
    img= img/0.5 - 1 #(-1,1)
    return img

train_set=tf.data.Dataset.from_tensor_slices(imgs_path)
train_set=train_set.map(preprocess).batch(batch_size)

def visualize():
    #imgs=next(iter(train_set))
    for imgs in train_set.take(1):
        print(imgs.shape)

        plt.figure(figsize=(10,10))
        for i in range(16):
            plt.subplot(4,4,i+1)
            plt.imshow((imgs[i]))
        plt.show()

##############################model################
#Generator tanh激活输出；Critic 线性激活输出
class Generator(Model):
    def __init__(self):
        super(Generator,self).__init__()
        #z:(b,100)>>>(b,3,3,512)>>>(b,64,64,3)
        self.fc=layers.Dense(3*3*512)
        self.act1 = layers.LeakyReLU()
        
        self.conv1 = layers.Conv2DTranspose(256,3,3,use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.act2 = layers.LeakyReLU()
        
        self.conv2 = layers.Conv2DTranspose(128,5,2,use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.act3 = layers.LeakyReLU()

        self.conv3 = layers.Conv2DTranspose(3,4,3,activation='tanh')

    def call(self,x,training=None):
        #(b,z)>>>(b,64,64,3)
        x=self.act1(self.fc(x))#(b,3*3*512)
        x=tf.reshape(x,shape=(-1,3,3,512))
        
        x=self.act2(self.bn1(self.conv1(x),training=training))
        x=self.act3(self.bn2(self.conv2(x),training=training))
        x=self.conv3(x)
        return x
        
class Critic(Model):
    '''discriminator,(b,h,w,c)>>>(b,1)'''
    def __init__(self):
        super(Critic,self).__init__()
        self.conv1=layers.Conv2D(64,5,strides=3,use_bias=False)#padding='valid
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.LeakyReLU()

        self.conv2=layers.Conv2D(128,5,strides=3,use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.act2 = layers.LeakyReLU()

        self.conv3=layers.Conv2D(256,5,strides=3,use_bias=False)
        self.bn3 = layers.BatchNormalization()
        self.act3 = layers.LeakyReLU()
        
        self.flat = layers.Flatten()#(b,h,w,c)>>>(b,-1)
        self.out = layers.Dense(1)#output 线性激活输出
        
    def call(self,inputs,training=None):
        #inputs:(b,64,64,3)>>>(b,-1)
        x=self.act1(self.bn1(self.conv1(inputs),training=training))
        x=self.act2(self.bn2(self.conv2(x),training=training))
        x=self.act3(self.bn3(self.conv3(x),training=training))
        
        x=self.flat(x)#(b,h,w,c)>>>(b,-1)
        logits=self.out(x)#(b,-1)>>>(b,1) without activation
        return logits

###############################training compile################
gen=Generator()
disc=Critic()
seed = tf.random.normal([batch_size,100],)#random seed

gen_opt = tf.keras.optimizers.Adam(2e-4)
disc_opt = tf.keras.optimizers.Adam(1e-4)#Discriminator学习率低一点，避免鉴别器too strong，生成器无法继续学习
cross_entropy=tf.keras.losses.BinaryCrossentropy(from_logits=True)
################loss#####################3
def gp_loss(disc,batch_x,fake_imgs):
    '''使用梯度惩罚的方法替代权值剪裁。需要满足函数在任意位置上的梯度都小于1，根据网络的输入来限制对应判别器的输出。对此我们更新目标函数，添加惩罚项。
        gradient penalty,batch_x:real img [b,h,w,c]
    '''
    t=tf.random.uniform([batch_x.shape[0],1,1,1])#shape:[b,1,1,1]插值系数(0~1)，进行一次采样
    inter=t*batch_x + (1-t)*fake_imgs #在真实数据分布和生成数据分布各进行一次采样，然后这两个分布上再做一次随机采样。它的范围是真实数据分布与生成数据分布中间的分布
    with tf.GradientTape() as tape:
        #若两个模型概率密度空间没有交集，kl divergence就会失效，梯度不会下降，
        #而Wasserstein距离可以解决这个问题
        tape.watch([inter])
        inter_logits = disc(inter)#将插值的图像输入给discriminator
    grads = tape.gradient(inter_logits,inter)#求偏导
    #grads:[b,h,w,c]>>> [b,-1]求二范数
    grads = tf.reshape(grads,[grads.shape[0],-1])
    gp = tf.norm(grads,axis=-1)#[b]
    gp = tf.reduce_mean((gp-1)**2)#求该范数与 1 的mse
    return gp

def gen_loss(fake_y):
    #generator优化器应当通过学习 将fake img判为1
    loss=cross_entropy(tf.ones_like(fake_y),fake_y)
    return tf.reduce_mean(loss)

def disc_loss(fake_y,real_y):
    real_loss=cross_entropy(tf.ones_like(real_y),real_y)#优化器应当将real img判为1
    fake_loss=cross_entropy(tf.zeros_like(fake_y),fake_y)#优化器应当将fake img判为0
    loss = real_loss+fake_loss
    return tf.reduce_mean(loss)

@tf.function
def train_gen_step(gen,disc,batch_z,is_training):
    with tf.GradientTape() as tape:
        fake_imgs = gen(batch_z,training=is_training)#生成fake imgs
        fake_y = disc(fake_imgs,training=is_training)#鉴别fake imgs，
        loss = gen_loss(fake_y)  #gen loss:generator生成图片cheat disc
    grads=tape.gradient(loss,gen.trainable_variables)
    gen_opt.apply_gradients(zip(grads,gen.trainable_variables))
    return loss     

@tf.function
def train_critic_step(gen,disc,batch_z,batch_x,is_training):
    '''WGAN training Critic,add GP loss(Gradient penalty)'''
    with tf.GradientTape() as tape:
        fake_imgs = gen(batch_z,training=is_training)#fake imgs with noise input

        fake_y=disc(fake_imgs,training=is_training)
        real_y=disc(batch_x,training=is_training)

        loss = disc_loss(fake_y,real_y)#critic loss
        gp = gp_loss(disc,batch_x,fake_imgs)#gradient penalty
        
        loss =loss + 5.*gp#添加gp loss
    grads=tape.gradient(loss,disc.trainable_variables)
    gen_opt.apply_gradients(zip(grads,disc.trainable_variables))

    return loss,gp

def train_fun(epochs,gen,disc,is_training=True):
    '''batch_z:noise input'''
    disc_losses,gen_losses,gp_losses=[],[],[]
    for epoch in range(epochs):
        for step,batch_x in enumerate(train_set):
            #给定的随机向量batch_z应符合（-1，1）均匀分布
            #if step%5==0:#每5个step训练一次discriminator
            batch_z = tf.random.uniform((batch_x.shape[0],100),minval=-1.,maxval=1.)#noise input
            #train discriminator first
            disc_loss,gp=train_critic_step(gen,disc,batch_z,batch_x,is_training)
            disc_losses.append(disc_loss)
            gp_losses.append(gp)
        #for step,batch_x in enumerate(train_set):
        #不能训练完一个epoch discriminator再训练gen,这时disc too strong
            #train generator 
            gen_loss= train_gen_step(gen,disc,batch_z,is_training)
            gen_losses.append(gen_loss)

        tf.print('current in epoch:{},\ndisc_loss:{},gen_loss:{},gp_loss:{}'.format(epoch,disc_loss.numpy(),gen_loss.numpy(),gp.numpy()))
        #可视化gen
        generate_and_save_images(gen,batch_z,epoch)
    
    return disc_losses,gen_losses,gp_losses   
        
def generate_and_save_images(model, test_input,epoch):
    # 注意 training` 设定为 False 因此，所有层都在推理模式下运行（batch_norm）。
    predictions = model(test_input, training=False)#(-1,1)
    #tf.print(tf.reduce_max(predictions),tf.reduce_min(predictions))
    predictions = (predictions+1.)*127.5 #pixel value:(0~255)
    fig = plt.figure(figsize=(4,4))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i])
        plt.axis('off')
    if epoch%10==0: #每10个epoch保存一次生成图片
        plt.savefig('image_at_{}_epoch.png'.format(epoch))
    plt.show()    

if __name__ == '__main__':
    
    disc_losses,gen_losses,gp_losses=train_fun(30,gen,disc)
    plt.plot(disc_losses,label='disc_loss')
    plt.plot(gen_losses,label='critic_loss')
    plt.plot(gp_losses,label='gp_losses')
    plt.legend()
    plt.show()

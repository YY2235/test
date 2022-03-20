import tensorflow as tf
import numpy as np
np.random.seed(1337)  # for reproducibility

import os
import sys
sys.path.append("../models")
sys.path.append("../base")
filename = os.path.basename(__file__)

from dbn import DBN
from cnn import CNN
from sup_sae import supervised_sAE
from base_func import run_sess
from tensorflow.examples.tutorials.mnist import input_data

# Loading dataset
# Each datapoint is a 8x8 image of a digit.
def read_data():
    def one_hot(Y):
        n=Y.shape[0]
        m=np.max(Y)+1
        oh_Y=np.zeros([n,m])
        for i in range(n):
            for j in range (n):
               if Y[i]==j:
                  oh_Y[i,j]=1
        return oh_Y
  
   # train_data=np.loadtxt('TYUT_jihe_18_21_train.csv',dtype=np.float32,delimiter=',') 
  #  test_data=np.loadtxt('TYUT_jihe_18_21_test.csv',dtype=np.float32,delimiter=',')
  
    #train_data=np.loadtxt('d:/E/shujuku/DBM feature extraction/sp_spec_datateagervmdtrain.csv',dtype=np.float32,delimiter=',') 
    #test_data=np.loadtxt('d:/E/shujuku/DBM feature extraction/sp_spec_datateagervmdtest.csv',dtype=np.float32,delimiter=',')
    
    #打乱顺序
    random_seed=15
    #data=np.loadtxt('d:/E/shujuku/BLquan/teagervmd/teagervmdqj.csv',dtype=np.float32,delimiter=',')
    #data=np.loadtxt('d:/E/shujuku/BLquan/mel图/melteagervmdqzlj.csv',dtype=np.float32,delimiter=',')
    data=np.loadtxt('d:/E/shujuku/DBM feature extraction/yundongshengxue/384j.csv',dtype=np.float32,delimiter=',')
#    data=np.loadtxt('d:/E/shujuku/DBM feature extraction/shunxu/melq+barkqj.csv',dtype=np.float32,delimiter=',')
   # data=np.loadtxt('d:/E/shujuku/SAE feature extraction/saver/melteagervmdqzlj.csv',dtype=np.float32,delimiter=',')
    #data=np.loadtxt('d:/E/shujuku/SAVEE/teagervmd/teagervmdqj.csv',dtype=np.float32,delimiter=',')
    shuffle_indices=np.random.permutation(np.arange(len(data)))
    data=data[shuffle_indices]
    #柏林
#    train_data=data[:200,:]
#    test_data=data[200:535,:]      
#yundongxue
    train_data=data[:1012,:]
    test_data=data[1012:1127,:]
    #SAVEE
    #train_data=data[:384,:]
    #test_data=data[384:480,:]
    #TYUT
    #train_data=data[:200,:]
    #test_data=data[200:250,:] 
    #RAVDESS1
    #train_data=data[:1152,:]
    #test_data=data[1152:1440,:]   
    
    X_train = train_data[:,:-1]
    Y_train = np.asarray(train_data[:,-1],dtype=np.int)
    
    X_test= test_data[:,:-1]
    Y_test= np.asarray(test_data[:,-1],dtype=np.int)
    Y_train = one_hot(Y_train)
    Y_test = one_hot(Y_test)
    return X_train , Y_train , X_test , Y_test,shuffle_indices

  
datasets=read_data()
X_train= datasets[0]
Y_train= datasets[1]
X_test= datasets[2] 
Y_test= datasets[3]  
shuffle_indices=datasets[4] 
#np.savetxt("../saver/deep_bolin_mel_200_150_indices.csv", shuffle_indices, fmt='%.4f',delimiter=",")
# Splitting data
datasets = [X_train, Y_train, X_test, Y_test]
#np.savetxt("../saver/deep_bolin_hb_200_150_indices.csv", shuffle_indices, fmt='%.4f',delimiter=",")
#np.savetxt("../saver1/deep_SAVEE_teagervmd_200_150_indices.csv", shuffle_indices, fmt='%.4f',delimiter=",")
#np.savetxt("../saver1/deep_SAVEE_hb_300_250_indices.csv", shuffle_indices, fmt='%.4f',delimiter=",")
np.savetxt("../yundongshengxue/indices.csv", shuffle_indices, fmt='%.4f',delimiter=",")
#np.savetxt("../shunxu/indices.csv", shuffle_indices, fmt='%.4f',delimiter=",")
#np.savetxt("../saver/deep_bolin_melteagervmd_400_150_indices.csv", shuffle_indices, fmt='%.4f',delimiter=",")
#np.savetxt("../saver2/deep_tyut_teagervmd_200_150_indices.csv", shuffle_indices, fmt='%.4f',delimiter=",")



x_dim=X_train.shape[1] #特征数
y_dim=Y_train.shape[1] #类别

tf.reset_default_graph()
# Training
select_case = 1

if select_case==1:
    classifier = DBN(
                 hidden_act_func='sigmoid',
                 output_act_func='softmax',
                 loss_func='cross_entropy', # gauss 激活函数会自动转换为 mse 损失函数
                 #struct=[x_dim, 40, 30, 15, y_dim],
#                 struct=[x_dim, 250, 225, 150, y_dim],
                 struct=[x_dim, 250,225,150, y_dim],
                 #合并
                 #struct=[x_dim, 300, 225, 150, y_dim],
                 lr=1e-4,
                 momentum=0.4,
                 use_for='classification',
                 bp_algorithm='rmsp',
                 epochs=1000,
                 batch_size=32,
                 #dropout=0.09,
                 units_type=['gauss','bin'],
                 rbm_lr=1e-3,
                 rbm_epochs=30,
                 cd_k=1,
                 pre_train=True)
if select_case==2:
    classifier = CNN(
                 output_act_func='softmax',
                 hidden_act_func='relu',
                 loss_func='cross_entropy',
                 use_for='classification',
                 lr=1e-3,
                 epochs=30,
                 img_shape=[p_dim,p_dim],
                 channels=[1, 6, 6, 64, y_dim], # 前几维给 ‘Conv’ ，后几维给 ‘Full connect’
                 layer_tp=['C','P','C','P'],
                 fsize=[[4,4],[3,3]],
                 ksize=[[2,2],[2,2]],
                 batch_size=32,
                 dropout=0.2)

if select_case==3:    
    classifier = supervised_sAE(           
                 output_func='softmax',
                 hidden_func='affine', # encoder：[sigmoid] | [affine] 
                 use_for='classification',
                 loss_func='mse', # decoder：[sigmoid] with ‘cross_entropy’ | [affine] with ‘mse’
                 struct=[784, 100, 100, 10],
                 lr=1e-4,
                 epochs=60,
                 batch_size=32,
                 dropout=0,
                 ae_type='dae', # ae | dae | sae
                 act_type=['sigmoid','affine'],
                 noise_type='mn', # Gaussian noise (gs) | Masking noise (mn)
                 beta=0.25,  # 惩罚因子权重（KL项 | 非噪声样本项）
                 p=0.5, # DAE：样本该维作为噪声的概率 / SAE稀疏性参数：期望的隐层平均活跃度（在训练批次上取平均）      
                 ae_lr=1e-3,
                 ae_epochs=20,
                 pre_train=True)       
run_sess(classifier,datasets,filename,load_saver='')
label_distribution = classifier.label_distribution
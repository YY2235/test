import tensorflow as tf
import numpy as np
np.random.seed(1337)  # for reproducibility

#import os
#import sys
#sys.path.append("../models")
#sys.path.append("../base")
#filename = os.path.basename(__file__)

from dbn import DBN
from cnn import CNN
from sup_sae import supervised_sAE
from base_func import run_sess
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler

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
    data=np.loadtxt('d:/E/shujuku/DBM feature extraction/yundongshengxue/yundongxuej.csv',dtype=np.float32,delimiter=',')
    label=np.loadtxt('d:/E/shujuku/DBM feature extraction/yundongshengxue/label.csv',dtype=np.int,delimiter=',')
#    data=np.loadtxt('d:/E/shujuku/DBM feature extraction/shunxu/melq+barkqj.csv',dtype=np.float32,delimiter=',')
   # data=np.loadtxt('d:/E/shujuku/SAE feature extraction/saver/melteagervmdqzlj.csv',dtype=np.float32,delimiter=',')
    #data=np.loadtxt('d:/E/shujuku/SAVEE/teagervmd/teagervmdqj.csv',dtype=np.float32,delimiter=',')
#    label = data1[:,-1]  
#    data = data1[:,:-1]
    
    shuffle_indices=np.random.permutation(np.arange(len(data)))
    data=data[shuffle_indices] 
    
    # 去除方差为0的列
    sel = VarianceThreshold(threshold=(0))  # .8 * (1 - .8)=0.159998
    data = sel.fit_transform(data)
    print("data") 
#    data = StandardScaler().fit_transform(data)
    data = MinMaxScaler().fit_transform(data)  # 特征归一化
        # 去除重复的列
    data = np.unique(data, axis=1)  

    
  

    #柏林
#    train_data=data[:200,:]
#    test_data=data[200:535,:]      
#yundongxue
    X_train=data[:1012,:]
    X_test=data[1012:1127,:]
    
    Y_train=label[:1012,]
    Y_test=label[1012:1127,]    
    
    
    #SAVEE
    #train_data=data[:384,:]
    #test_data=data[384:480,:]
    #TYUT
    #train_data=data[:200,:]
    #test_data=data[200:250,:] 
    #RAVDESS1
    #train_data=data[:1152,:]
    #test_data=data[1152:1440,:]   
    
#    X_train = train_data[:,:-1]
#    Y_train = np.asarray(train_data[:,-1],dtype=np.int)
# 
#    
#    
#    X_test= test_data[:,:-1]
#    Y_test= np.asarray(test_data[:,-1],dtype=np.int)
    
    Y_train = one_hot(Y_train)
    Y_test = one_hot(Y_test)
#    return  data,label,X_train,Y_train , X_test , Y_test,shuffle_indices

  
datasets=read_data()
X_train= datasets[0]
Y_train= datasets[1]
X_test= datasets[2] 
Y_test= datasets[3]  
shuffle_indices=datasets[4] 
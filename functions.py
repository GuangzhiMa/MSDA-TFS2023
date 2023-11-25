from torch.autograd import Function
import numpy as np
import torch
import time

from sklearn.model_selection import train_test_split

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

def Encryption(letterfuture, letterlabels, encry, y_number):
    a = letterfuture[:,y_number].min()
    b = letterfuture[:,y_number].max()
    c = (b-a)/encry
    ar, number = np.unique(letterlabels, return_counts = True) 
    class_number = ar.shape[0]
    feature_number = letterfuture.shape[1]

    number_i = []
    number_k = []
    for i in range(encry):
        for k in range(class_number):
            t1 = np.where(letterfuture[:,y_number]< (a+c*i))
            t2 = np.where(letterfuture[:,y_number]< (a+c*(i+1)))  
            t3 = np.where(letterlabels == k )
            t01 = list(set(t1[0]).intersection(set(t3[0])))
            t02 = list(set(t2[0]).intersection(set(t3[0])))
            t = list(set(t02).difference(set(t01)))
            if len(t) > 0: 
                number_i.append(i)
                number_k.append(k)

    letterencry = np.zeros((len(number_i),feature_number*2 + 1))


    for i in range(len(number_i)):
        for j in range(int(letterencry.shape[1] / 2)):
            t1 = np.where(letterfuture[:,y_number]< (a+c*number_i[i]))
            t2 = np.where(letterfuture[:,y_number]< (a+c*(number_i[i]+1)))  
            t3 = np.where(letterlabels == number_k[i] )
            t01 = list(set(t1[0]).intersection(set(t3[0])))
            t02 = list(set(t2[0]).intersection(set(t3[0])))
            t = list(set(t02).difference(set(t01)))
            letter0 = letterfuture[t, :]
            letterencry[i][2*j] = letter0[:,j].min()
            letterencry[i][2*j + 1] = letter0[:,j].max()
        letterencry[i][-1] = number_k[i] 
            
    letterencry_labels = letterencry[:,-1]              
    letterencry_feature = letterencry[:,0:32] 
    
    return letterencry_feature, letterencry_labels

def letterencry(letterfuture, letterlabels, encry, k):
    letterencry_feature, letterencry_labels = Encryption(letterfuture, letterlabels, encry, 0)
    letterencry_labels = letterencry_labels.reshape(letterencry_labels.shape[0], 1)
    if k>1:
        for i in range(k-1):
            letterencry_feature1, letterencry_labels1 = Encryption(letterfuture, letterlabels, encry, i + 1)
            letterencry_feature = np.vstack((letterencry_feature, letterencry_feature1))
            letterencry_labels = np.vstack((letterencry_labels, letterencry_labels1.reshape(letterencry_labels1.shape[0], 1)))
    return letterencry_feature, letterencry_labels
   

#split dataset
def Split_dataset(data, label, vali_size, test_size, random_state):
    letter_train, letter_vali, y_train, y_vali = train_test_split(data, label, test_size=vali_size, random_state=random_state)   
    letter_train, letter_test, y_train, y_test = train_test_split(letter_train, y_train, test_size=test_size, random_state=random_state)
    return letter_train, letter_vali, letter_test, y_train, y_vali, y_test

#Gaussian fuzzy 
def data_sigma(class_number, future, encry_feature, encry_label):
    letterencry_sigma = np.zeros((encry_feature.shape[0], future.shape[1]*2 ))   

    for i in range(class_number):
        tt = np.where(encry_label == i)
        for j in range(future.shape[1]):
            left = encry_feature[tt, 2*j]
            sigma1 = np.std(left)
            right = encry_feature[tt, 2*j+1]
            sigma2 = np.std(right)
            for k in range(tt[0].shape[0]):
                letterencry_sigma[tt[0][k], 2*j] = sigma1
                letterencry_sigma[tt[0][k], 2*j+1] = sigma2
    return letterencry_sigma


def DFdataset(letterencry_feature, letterencry_labels, letterencry_sigma, beta, feature_number, batch_size, vali_size, test_size, random_state):
    #MOM
    encry_number = letterencry_labels.shape[0]
    letterencry_MOM = np.array([])
    for i in range(feature_number):
        M = beta*letterencry_feature[:,2*i] + (1-beta)*letterencry_feature[:,2*i+1]
        letterencry_MOM = np.concatenate((letterencry_MOM, M))
    letterencry_MOM = (letterencry_MOM.reshape(feature_number, encry_number)).T  
    MOM_train, MOM_vali, MOM_test, y1_train, y1_vali, y1_test = Split_dataset(letterencry_MOM, letterencry_labels, vali_size, test_size, random_state)
    train1_dataset = torch.utils.data.TensorDataset(torch.tensor(MOM_train, dtype=torch.float32), torch.tensor(y1_train).long())
    train1_iter = torch.utils.data.DataLoader(train1_dataset, batch_size)
    vali1_dataset = torch.utils.data.TensorDataset(torch.tensor(MOM_vali, dtype=torch.float32), torch.tensor(y1_vali).long())
    vali1_iter = torch.utils.data.DataLoader(vali1_dataset, batch_size)
    test1_dataset = torch.utils.data.TensorDataset(torch.tensor(MOM_test, dtype=torch.float32), torch.tensor(y1_test).long())
    test1_iter = torch.utils.data.DataLoader(test1_dataset, batch_size)
         
    #COG
    letterencry_COG = np.array([])
    for i in range(feature_number):
        M = (beta/2 + 1/4)*letterencry_feature[:,2*i] + (3/4 - beta/2)*letterencry_feature[:,2*i+1]
        letterencry_COG = np.concatenate((letterencry_COG, M))
    letterencry_COG = (letterencry_COG.reshape(feature_number, encry_number)).T 
    COG_train, COG_vali, COG_test, y2_train, y2_vali, y2_test = Split_dataset(letterencry_COG, letterencry_labels, vali_size, test_size, random_state)
    train2_dataset = torch.utils.data.TensorDataset(torch.tensor(COG_train, dtype=torch.float32), torch.tensor(y2_train).long())
    train2_iter = torch.utils.data.DataLoader(train2_dataset, batch_size)
    vali2_dataset = torch.utils.data.TensorDataset(torch.tensor(COG_vali, dtype=torch.float32), torch.tensor(y2_vali).long())
    vali2_iter = torch.utils.data.DataLoader(vali2_dataset, batch_size)
    test2_dataset = torch.utils.data.TensorDataset(torch.tensor(COG_test, dtype=torch.float32), torch.tensor(y2_test).long())
    test2_iter = torch.utils.data.DataLoader(test2_dataset, batch_size)
         

    #ALC
    letterencry_ALC = np.array([])
    for i in range(feature_number):
        M = (beta/3 + 1/3)*letterencry_feature[:,2*i] + (2/3 - beta/3)*letterencry_feature[:,2*i+1]
        letterencry_ALC = np.concatenate((letterencry_ALC, M))
    letterencry_ALC = (letterencry_ALC.reshape(feature_number, encry_number)).T
    ALC_train, ALC_vali, ALC_test, y3_train, y3_vali, y3_test = Split_dataset(letterencry_ALC, letterencry_labels, vali_size, test_size, random_state)
    train3_dataset = torch.utils.data.TensorDataset(torch.tensor(ALC_train, dtype=torch.float32), torch.tensor(y3_train).long())
    train3_iter = torch.utils.data.DataLoader(train3_dataset, batch_size)
    vali3_dataset = torch.utils.data.TensorDataset(torch.tensor(ALC_vali, dtype=torch.float32), torch.tensor(y3_vali).long())
    vali3_iter = torch.utils.data.DataLoader(vali3_dataset, batch_size)
    test3_dataset = torch.utils.data.TensorDataset(torch.tensor(ALC_test, dtype=torch.float32), torch.tensor(y3_test).long())
    test3_iter = torch.utils.data.DataLoader(test3_dataset, batch_size)
         
    #VAL
    letterencry_VAL = np.array([])
    for i in range(feature_number):
        M = (2*beta/3 + 1/6)*letterencry_feature[:,2*i] + (5/6 - 2*beta/3)*letterencry_feature[:,2*i+1]
        letterencry_VAL = np.concatenate((letterencry_VAL, M))
    letterencry_VAL = (letterencry_VAL.reshape(feature_number, encry_number)).T
    VAL_train, VAL_vali, VAL_test, y4_train, y4_vali, y4_test = Split_dataset(letterencry_VAL, letterencry_labels, vali_size, test_size, random_state)  
    train4_dataset = torch.utils.data.TensorDataset(torch.tensor(VAL_train, dtype=torch.float32), torch.tensor(y4_train).long())
    train4_iter = torch.utils.data.DataLoader(train4_dataset, batch_size)
    vali4_dataset = torch.utils.data.TensorDataset(torch.tensor(VAL_vali, dtype=torch.float32), torch.tensor(y4_vali).long())
    vali4_iter = torch.utils.data.DataLoader(vali4_dataset, batch_size)
    test4_dataset = torch.utils.data.TensorDataset(torch.tensor(VAL_test, dtype=torch.float32), torch.tensor(y4_test).long())
    test4_iter = torch.utils.data.DataLoader(test4_dataset, batch_size)
    
    #Gaussian COG
    letterencry_GauCOG = np.array([])
    for i in range(feature_number):
        M = beta*letterencry_feature[:,2*i] + (1-beta)*letterencry_feature[:,2*i+1] - letterencry_sigma[:, 2*i]**2 + letterencry_sigma[:, 2*i+1]**2
        letterencry_GauCOG = np.concatenate((letterencry_GauCOG, M))
    letterencry_GauCOG = (letterencry_GauCOG.reshape(feature_number, encry_number)).T  
    GauCOG_train, GauCOG_vali, GauCOG_test, y5_train, y5_vali, y5_test = Split_dataset(letterencry_GauCOG, letterencry_labels, vali_size, test_size, random_state)
    train5_dataset = torch.utils.data.TensorDataset(torch.tensor(GauCOG_train, dtype=torch.float32), torch.tensor(y5_train).long())
    train5_iter = torch.utils.data.DataLoader(train5_dataset, batch_size)
    vali5_dataset = torch.utils.data.TensorDataset(torch.tensor(GauCOG_vali, dtype=torch.float32), torch.tensor(y5_vali).long())
    vali5_iter = torch.utils.data.DataLoader(vali5_dataset, batch_size)
    test5_dataset = torch.utils.data.TensorDataset(torch.tensor(GauCOG_test, dtype=torch.float32), torch.tensor(y5_test).long())
    test5_iter = torch.utils.data.DataLoader(test5_dataset, batch_size)
    
    #Gaussian ALC
    letterencry_GauALC = np.array([])
    for i in range(feature_number):
        M = beta*letterencry_feature[:,2*i] + (1-beta)*letterencry_feature[:,2*i+1] - letterencry_sigma[:, 2*i]/2 + letterencry_sigma[:, 2*i+1]/2
        letterencry_GauALC = np.concatenate((letterencry_GauALC, M))
    letterencry_GauALC = (letterencry_GauALC.reshape(feature_number, encry_number)).T  
    GauALC_train, GauALC_vali, GauALC_test, y6_train, y6_vali, y6_test = Split_dataset(letterencry_GauALC, letterencry_labels, vali_size, test_size, random_state)
    train6_dataset = torch.utils.data.TensorDataset(torch.tensor(GauALC_train, dtype=torch.float32), torch.tensor(y6_train).long())
    train6_iter = torch.utils.data.DataLoader(train6_dataset, batch_size)
    vali6_dataset = torch.utils.data.TensorDataset(torch.tensor(GauALC_vali, dtype=torch.float32), torch.tensor(y6_vali).long())
    vali6_iter = torch.utils.data.DataLoader(vali6_dataset, batch_size)
    test6_dataset = torch.utils.data.TensorDataset(torch.tensor(GauALC_test, dtype=torch.float32), torch.tensor(y6_test).long())
    test6_iter = torch.utils.data.DataLoader(test6_dataset, batch_size)
    
    #Gaussian VAL
    letterencry_GauVAL = np.array([])
    for i in range(feature_number):
        M = beta*letterencry_feature[:,2*i] + (1-beta)*letterencry_feature[:,2*i+1] - letterencry_sigma[:, 2*i] + letterencry_sigma[:, 2*i+1]
        letterencry_GauVAL = np.concatenate((letterencry_GauVAL, M))
    letterencry_GauVAL = (letterencry_GauVAL.reshape(feature_number, encry_number)).T  
    GauVAL_train, GauVAL_vali, GauVAL_test, y7_train, y7_vali, y7_test = Split_dataset(letterencry_GauVAL, letterencry_labels, vali_size, test_size, random_state)
    train7_dataset = torch.utils.data.TensorDataset(torch.tensor(GauVAL_train, dtype=torch.float32), torch.tensor(y7_train).long())
    train7_iter = torch.utils.data.DataLoader(train7_dataset, batch_size)
    vali7_dataset = torch.utils.data.TensorDataset(torch.tensor(GauVAL_vali, dtype=torch.float32), torch.tensor(y7_vali).long())
    vali7_iter = torch.utils.data.DataLoader(vali7_dataset, batch_size)
    test7_dataset = torch.utils.data.TensorDataset(torch.tensor(GauVAL_test, dtype=torch.float32), torch.tensor(y7_test).long())
    test7_iter = torch.utils.data.DataLoader(test7_dataset, batch_size)
    
    train_iter = [train1_iter, train2_iter, train3_iter, train4_iter, train5_iter, train6_iter, train7_iter]
    vali_iter = [vali1_iter, vali2_iter, vali3_iter, vali4_iter, vali5_iter, vali6_iter, vali7_iter]
    test_iter = [test1_iter, test2_iter, test3_iter, test4_iter, test5_iter, test6_iter, test7_iter]
    
    return train_iter, vali_iter, test_iter 















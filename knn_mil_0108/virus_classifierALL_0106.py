# -*- coding: utf-8 -*-
"""
Dec 29
@yz
Created on Thu Nov 24 14:15:25 2022
@author: caixd
"""

import math
import numpy as np
import gensim
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

'''
model = gensim.models.Word2Vec.load("genome.w2vmodel")

vec = model.wv['ATAATCGT']
print(model.wv.most_similar(positive=['ATAATATT'], topn=5))
'''

def seq2sent2vec(filename, readlen, readnum, k, wvmodel):
    #convert  sequences or reads to sentences, each word to a vector 
    #return a 3D numpy array of readnum x (readlen-k+1) x 100
    #filename: the file that contain sequences (reads)
    #readlen: length of reads (should be <=500, sequence lenth in the file)
    #readnum: number of reads, should be less than the number of reads in the file (5000)
    #k: k-mer as a word  
    #wvmodel: Word2Vec model
    nwords=readlen-k+1
    datavec=np.zeros([readnum,nwords,100])
    with open(filename, 'r' ) as freads:
        for i, line in enumerate(freads):
            temp=line.strip()
            for j in range(nwords):
                word=temp[j:j+k]
                temp2=wvmodel.wv[word]
                #temp2.shape=(1,len(temp2))
                datavec[i,j,:]=temp2

            if i >= readnum-1:
                break  
    return datavec



readlen=300
k=8
readnum_training = 5000 #number of reads from each of human virus and non-human virus
readnum_testing = 1000
readnum = readnum_training + readnum_testing
nwords = readlen-k+1  #numer of words per sentences 
wvmodel = gensim.models.Word2Vec.load("genome.w2vmodel")    

#get the data
filename="SampleVirusReadsH.txt"
data1=seq2sent2vec(filename, readlen, readnum, k, wvmodel)
filename="SampleVirusReadsNH.txt"
data2=seq2sent2vec(filename, readlen, readnum, k, wvmodel)

#reshape 3D array data1 and data2 into 2D array as training data, each row is 1x100 vector as a sample
xtrain=np.concatenate((data1[0:readnum_training].reshape(-1,100), data2[0:readnum_training].reshape(-1,100)),axis=0)
temp=readnum_training*nwords
ytrain = np.concatenate((np.zeros(temp), np.ones(temp)))
#ytrain = np.concatenate((np.ones(temp), np.zeros(temp)))

print(xtrain.shape)

#testing data is still in 3D arrary
temp=readnum_training+readnum_testing
xtest = np.concatenate((data1[readnum_training:temp],data2[readnum_training:temp]), axis=0)
ytestread = np.concatenate((np.zeros(readnum_testing), np.ones(readnum_testing)))       
#ytestread = np.concatenate((np.ones(readnum_testing), np.zeros(readnum_testing)))       


# del data1, data2

print(xtest.shape)

#train the classifier
clf = LogisticRegression(solver='liblinear', random_state=0)
#clf = LogisticRegression(penalty='l2',solver='newton-cg',max_iter=200)
clf.fit(xtrain,ytrain)

# clf = RandomForestClassifier(n_estimators=1000)
# clf.fit(xtrain, ytrain)

#compute training error
ypred=clf.predict(xtrain)
ypredread=np.zeros(2*readnum_training)
for i in range(2*readnum_training):
    temp=ypred[i*nwords:(i+1)*nwords]
    if np.sum(temp)> nwords/2:
        ypredread[i]=1
yread=np.concatenate((np.zeros(readnum_training),np.ones(readnum_training)))
        
cnf_matrix_train = confusion_matrix(yread, ypredread)
print(cnf_matrix_train)

#compute testing error
ypredread=np.zeros(2*readnum_testing)
for i in range(2*readnum_testing):
    ypred=clf.predict(xtest[i])
    if np.sum(ypred) > nwords/2:
        ypredread[i] = 1

cnf_matrix_test = confusion_matrix(ytestread, ypredread, labels = [0,1])
print(cnf_matrix_test)


#compute testing error
ypredcount0=[]
ypredcount1=[]
for i in range(2*readnum_testing):
    ypred=clf.predict(xtest[i])
    if i < readnum_testing:
        ypredcount0.append(np.sum(ypred))
    else:
        ypredcount1.append(np.sum(ypred))

n, bins, patches = plt.hist(ypredcount0, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

'''
clf = svm.SVC(kernel='linear')
clf.fit(xtrain, ytrain)
#y=clf.predict(x)


cov1=covb
print(np.linalg.matrix_rank(cov1))
print(np.linalg.cond(cov1))
evalue, evector = np.linalg.eig(cov1)
print(evalue)
'''
import argparse
import os
import numpy as np
#from sklearn.externals import joblib
import pandas as pd
from azureml.core.run import Run
from azureml.core import Workspace, Dataset

#Import data and understanding it.

import collections
import logging
import os
import pathlib
import re
import string
import sys
import time
from collections import Counter
import re
from itertools import combinations

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf
import json
from sklearn.feature_extraction.text import CountVectorizer

# Function to create character three_gram  
def three_gram(a):
  ans=[]
  for i in a:
    tmp=[]
    for j in i:
      n=len(j)
      if n>2:
        for k in range(n-2):
          tmp.append(j[k:k+3])
    ans.append(tmp)
  return ans


def create_vocab(data):
  df=data.copy()
  a=df.groupby('1')
  d={}
  for i,g in a:
    d[i]=g['0'].tolist()

  for i in range(8):
    # Lower case the strings
    d[i]=[x.lower() for x in d[i]]
    
    # Remove spacial character and numbers
    d[i]=[re.sub('\W+',' ', x) for x in d[i]]
    d[i]=[re.sub(r'\d+', '', x) for x in d[i]]
    
    # strip the strings
    d[i]=list(map(str.strip,d[i]))

    #split the strings
    d[i]=[x.split() for x in d[i]]

  # Create a vocabulary of 1st 250 character 3-gram 

  # create a counter
  count_vec={}
  for i in range(8):
    x=three_gram(d[i])
    c=Counter()
    x=[Counter(k) for k in x]
    for j in x:
      c=c+j
    count_vec[i]=c
  # sort the keys and vocab
  voc={}
  vocab=[]
  for i in range(8):
    count_vec[i]={k: v for k, v in sorted(count_vec[i].items(), key=lambda item: item[1],reverse=True)}
    l=list(count_vec[i].keys())[:200]
    vocab=vocab+l
    voc[i]=l
  vocab=list(set(vocab))
  return vocab


def feature_extraction(data):
  #group the data by languages
  trn=data.copy()
  a=trn.groupby('1')
  ftr={}
  vocab=create_vocab(trn)
  vectorizer=CountVectorizer(analyzer='char',ngram_range=(3,3),vocabulary=vocab)
  for i,g in a:
    x=list(set(g['0'].tolist()))
    ftr[i]=vectorizer.fit_transform(x).toarray()
  for i in range(8):
    ftr[i]=ftr[i][~np.all(ftr[i]==0,axis=1)]

# Combine all the arrays

  # create labels
  lbl={}
  ftr_={}
  for i in range(8):
    lbl[i]=np.array([i]*ftr[i].shape[0]).reshape((ftr[i].shape[0],1))
    ftr_[i]=np.append(ftr[i],lbl[i],1)
    if i==0:
      train=ftr_[i]
    else:
      train=np.append(train,ftr_[i],0)

  #Shuffle the dataset
  np.random.shuffle(train)

  trainx=train[:,:-1]
  trainy=train[:,-1]

  return trainx,trainy

def DNN_model(n,nlayers,nodes):
  inp=tf.keras.Input(shape=(n))
  l=tf.keras.layers.Dense(512,activation='relu')(inp)
  for i in range(nlayers):
    l=tf.keras.layers.Dense(nodes,activation='relu')(l)
  out=tf.keras.layers.Dense(8,activation='softmax')(l)
  model=tf.keras.Model(inputs=inp,outputs=out)

  ls=tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True)
  model.compile(optimizer='Adam',loss=ls,metrics=['sparse_categorical_accuracy'])
  return model

def main():
  # Add arguements
  parser= argparse.ArgumentParser()
  parser.add_argument('--path', type=str, dest='path', default='trn.csv', help='mounting data folder')
  parser.add_argument('--nlayers',type=int,default=3,help='Number of hidden layers')
  parser.add_argument('--nodes',type=int,default=128,help='Number of Nodes to hidden layers')

  args=parser.parse_args()

  run.log('Number of layers:',np.int(args.nlayers))
  run.log('Number of Nodes:',np.int(args.nodes))

  # Load dataset
  path=args.path
  df=pd.read_csv(path)

  x,y=feature_extraction(df)

  model=DNN_model(n=x.shape[1],nlayers=args.nlayers,nodes=args.nodes)

  h=model.fit(x,y,batch_size=1024,epochs=10,validation_split=0.15)

  #os.makedirs('outputs', exist_ok=True)
  #joblib.dump(model, 'outputs/model.joblib')

  run.log('Accuracy',np.float(h.history['val_sparse_categorical_accuracy'][-1]))


run = Run.get_context()
if __name__ == '__main__':
  main()
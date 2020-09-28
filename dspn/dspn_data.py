# -*- coding: utf-8 -*-


import torch
import torch.utils.data

import numpy as np

import sys



def get_loader(dataset, batch_size, num_workers=1, shuffle=False):
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
    )


class TestDataset(torch.utils.data.Dataset):
    def __init__(self,pair_path, size, batch_size, lineNums , vNum, train=True, full=False, max_size=None):
        self.train = train
        self.full = full
        self.size=size
        self.max_size=max_size
        self.data = self.cache(pair_path, lineNums,vNum)
        #self.max = 20
        self.batch_size=batch_size
        

    def cache(self, pair_path,lineNums, vNum):
        print("Processing dataset...")
        sample_data, max_size = extract_training_sets(pair_path,self.size,lineNums)
        if self.max_size == None:
            self.max_size=max_size 
        data = []
        #print(len(sample_data))
        for datapoint in sample_data:
            set_train, label = datapoint
            set_train = [int(i) for i in set_train]
            label = [int(i) for i in label]
            
            
            #if not self.train:
                #set_train.sort()
                #print("set_train {}".format(set_train))
            
            set_train = one_hot_encoder(np.array(set_train), vNum ,max_size).transpose()
            #print("set_train.shape {}".format(set_train.shape))

            #set_train = [np.array(set_train).transpose()]
            #print(set_train.shape)
            label = one_hot_encoder(np.array(label), vNum,max_size).transpose()
            
            #if not self.train:
                #set_train.sort()
                #print("set_train.shape {}".format(set_train.shape))
                #label.sort()
                #print(label)
                #a=utils.one_hot_to_number(torch.tensor(set_train))
                #a.sort()
                #print(a)
                #print("-------")
            #print("label shape {}".format(label.shape))
            #label = np.array(label).transpose()
            data.append((torch.Tensor(set_train), torch.Tensor(label), label.shape[0]))
            #torch.save(data, cache_path)
        print("Done!")
        return data
    

    def __getitem__(self, item):
        sys.exit("__getitem__ not implemented")


    def __len__(self):
        return self.size

def extract_training_sets(filename, size,lineNums):
  content, max_size = load_file(filename,size,lineNums)
  
  X = [x for i,x in enumerate(content) if i%2==0]
  y = [x for i,x in enumerate(content) if i%2==1]
  
  # Transforming data format
  X = [i.split() for i in X]
  y = [i.split() for i in y]

  '''
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42)
  ''' 
  return list(zip(X,y)), max_size


def load_file(filename,size,lineNums):
    max_size=0
    with open(filename) as f:
        content=[]
        lineNum = 0
        while len(lineNums)>0:
            line = f.readline() 
            if not line: 
                break 
            if lineNum != lineNums[0]:
                lineNum += 1 
            else:
                if len(line.split())>max_size:
                    max_size=len(line.split())
                    
                content.append(line)
                lineNum += 1                   
                line = f.readline()
                content.append(line)
                lineNum += 1 
                
                
                lineNums=np.delete(lineNums, 0)          
                
  
    return content, max_size


def one_hot_encoder(data, max_value, max_size):
    shape = (max_size, max_value)
    one_hot = np.zeros(shape)
    rows = np.arange(data.size)
    one_hot[rows, data] = 1
    
    #print(one_hot.shape)
    #print(one_hot)
    #input("wait")
    
    return one_hot
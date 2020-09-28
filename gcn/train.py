from __future__ import division
from __future__ import print_function

import os
import argparse
import numpy as np


import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split

import util as utils
from models import GCN
import sys
from stratLearner import InputInstance


# Training settings
parser = argparse.ArgumentParser()


parser.add_argument(
    '--dataname',  default='kro', 
                    choices=['kro, power768, ER512'])
parser.add_argument(
    '--vNum', type=int, default=1024, choices=[1024,768,512],
                    help='kro 1024, power768 768, ER512 512')
    
parser.add_argument(
    '--trainNum', type=int, default=270, help='number of training data') 
parser.add_argument(
    '--valNum', type=int, default=270, help='number of validation data')   
parser.add_argument(
    '--testNum', type=int, default=270, help='number of testing data')   

parser.add_argument(
    '--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')

parser.add_argument(
    '--thread', type=int, default=1, help='number of threads')

parser.add_argument(
    '--output', action="store_true", help='if output prediction')


parser.add_argument(
    '--seed', type=int, default=42, help='Random seed.')
parser.add_argument(
    '--epochs', type=int, default=10,
                    help='Number of epochs to train.')
parser.add_argument(
    '--lr', type=float, default=1e-3,
                    help='Initial learning rate.')
parser.add_argument(
    '--weight_decay', type=float, default=1e-2,
                    help='Weight decay (L2 loss on parameters).')

parser.add_argument(
    '--dropout', type=float, default=0.4,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument(
    "--train-only", action="store_true", default=False,
                    help="Only run training, no evaluation")
parser.add_argument(
    "--eval-only", action="store_true", 
                    help="Only run evaluation, no training")
parser.add_argument(
    "--batch-size", default=10,
                    help="Batch size of the training/testing set")


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    
dataname=args.dataname
vNum = args.vNum

trainNum =args.trainNum
valNum =args.valNum
testNum =args.testNum
pairMax=2500
totalNum=trainNum+valNum+testNum




torch.set_num_threads(args.thread)




#simulation times, small number for testing
infTimes = 10
    
#get data
path = os.getcwd() 
data_path=os.path.abspath(os.path.join(path, os.pardir))+"/data"
pair_path = "{}/{}/{}_pair_{}".format(data_path,dataname,dataname,pairMax)
graphPath = "{}/{}/{}_diffusionModel".format(data_path,dataname,dataname)



instance = InputInstance(graphPath, None, 0, vNum, 0, None, 
                         None, None)


print('Building dataset...')
dataset = utils.TestDataset(pair_path, graphPath, vNum, totalNum, train=True, dsize=True)
print(len(dataset.data))
train_dataset, val_dataset, test_dataset = random_split(dataset, (trainNum, valNum, testNum))
#train_dataset, test_dataset = random_split(train_dataset, (len(train_dataset) - len(train_dataset)//5, len(train_dataset)//5))

# Getting data loaders for training, validation, and testing
train_loader = utils.get_loader(train_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True)
valid_loader = utils.get_loader(val_dataset,  batch_size=valNum, num_workers=0, shuffle=False)
test_loader = utils.get_loader(test_dataset, batch_size=testNum, num_workers=0, shuffle=False)

data_loaders = {"train": train_loader, "val": valid_loader}



# Model and optimizer
model = GCN(vNum, dropout=args.dropout)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()


def train(model, loader, optimizer, epoch=10):
    # Each epoch has a training and validation phase
    print("Epoch: {}".format(epoch))
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
            torch.set_grad_enabled(True)
            loader = data_loaders[phase]
            iters_per_epoch = len(loader)
        else:
            model.eval()  # Set model to evaluate mode
            loader = data_loaders[phase]
            iters_per_epoch = len(loader)

        for i, sample in enumerate(loader, start=epoch * iters_per_epoch):
            if args.cuda:
                input, target, adj, card = map(lambda x: x.cuda(), sample)
            else:
                input, target, adj, card = map(lambda x: x, sample)
    
            prediction = []
            for item in range(len(input)):
                result = model(input[item], adj[item])
                prediction.append(result)

    
            prediction = torch.stack(prediction, dim=0)
            loss = F.binary_cross_entropy_with_logits(prediction, target.squeeze())

        
    
            if phase == "train":
                optimizer.zero_grad() # zero the parameter (weight) gradients
                loss.backward() # backward pass to calculate the weight gradients
                optimizer.step() # update the weights
            else:
                output=loss.item()
            
                
        print("{} Losses: {:.04f}".format(phase, loss))
        
    return output
    
            
            
def test(model, loader, optimizer, epoch=0):
    model.eval()  # Set model to evaluate mode
    loader =loader
    iters_per_epoch = len(loader)
    
    true_export, pred_export, true_import = [], [], []
    
    for i, sample in enumerate(loader, start=epoch * iters_per_epoch):
            if args.cuda:
                input, target, adj, card = map(lambda x: x.cuda(), sample)
            else:
                input, target, adj, card = map(lambda x: x, sample)
    
            prediction = []
            for item in range(len(input)):
                result = model(input[item], adj[item])
                prediction.append(result)
                #results.append(result)
    
            prediction = torch.stack(prediction, dim=0)
            
            # calculate the loss between predicted and target keypoints
            loss = F.binary_cross_entropy_with_logits(prediction, target.squeeze())
            print("\n{} Loss: {}".format('Test', loss.item()))
            
            for pre, tru, inp, c in zip(prediction, target, input, card):
                #print("pre {}".format(pre.shape))
                #print("tru {}".format(tru.shape))
                #print("inp {}".format(inp.shape))
                k=int(sum(inp.sum(dim=1)).item())
               # print(k)
                true_import.append(torch.topk(inp.sum(dim=1), k).indices.numpy())
                pred_export.append(torch.topk(pre.cpu().detach(), k).indices.numpy())
                true_export.append(torch.topk(tru.squeeze(), k).indices.numpy())
                
    X_test=[]
    Y_test=[]
    Y_pred=[]
    for y_pred, y_test, x_test in zip(pred_export, true_export, true_import):
        X_test.append(utils.list_to_set(x_test))
        Y_test.append(utils.list_to_set(y_test))
        Y_pred.append(utils.list_to_set(y_pred))

    utils.testing(X_test, Y_test, Y_pred, instance, args, infTimes=1080)
            
            
# Training the model
t_loss=sys.maxsize
for k in range(args.epochs):   
    c_loss=train(model, data_loaders, optimizer, epoch=k)
    #print(t_loss)
    if c_loss>t_loss:
        break
    else:
        #print("********")
        t_loss=c_loss
        #print(t_loss)
# Generating testing report
if not args.train_only:
    test(model, test_loader, optimizer, epoch=k)



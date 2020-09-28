# -*- coding: utf-8 -*-

import os
import argparse


import torch
import torch.nn.functional as F




import numpy as np
import matplotlib

matplotlib.use("Agg")


import dspn_data as data
import model
import dspn_util as utils
import sys

from dspn_method import InputInstance




parser = argparse.ArgumentParser()

# generic params
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

parser.add_argument("--resume", help="Path to log file to resume from")
#parser.add_argument("--resume", default="logs/2020-04-22_14_15_39", help="Path to log file to resume from")
parser.add_argument("--encoder", default="MLPEncoder", help="Encoder")
parser.add_argument("--decoder", default="DSPN", help="Decoder")
parser.add_argument(
    "--epochs", type=int, default=3, help="Number of epochs to train with"
)
parser.add_argument(
    "--latent", type=int, default=32, help="Dimensionality of latent space"
)
parser.add_argument(
    "--dim", type=int, default=512, help="Dimensionality of hidden layers"
)
parser.add_argument(
    "--lr", type=float, default=1e-2, help="Outer learning rate of model"
)
parser.add_argument(
    "--batch-size", type=int, default=270, help="Batch size to train with"
)
parser.add_argument(
    "--num-workers", type=int, default=1, help="Number of threads for data loader"
)
parser.add_argument(
    "--dataset",
    choices=["mnist", "clevr-box", "clevr-state"],
    help="Use MNIST dataset",
)
parser.add_argument(
    "--no-cuda",
    default=True,
    action="store_true",
    help="Run on CPU instead of GPU (not recommended)",
)
parser.add_argument(
    "--train-only", action="store_true", help="Only run training, no evaluation"
)
parser.add_argument(
    "--eval-only", action="store_true", help="Only run evaluation, no training"
)
parser.add_argument("--multi-gpu", default=False, action="store_true", help="Use multiple GPUs")
parser.add_argument(
    "--show", action="store_true", help="Plot generated samples in Tensorboard"
)

parser.add_argument("--supervised", default=True, action="store_true", help="")
parser.add_argument("--baseline", action="store_true", help="Use baseline model")

parser.add_argument("--export-dir", default='pred', type=str, help="Directory to output samples to")
parser.add_argument(
    "--export-n", type=int, default=10 ** 9, help="How many samples to output"
)
parser.add_argument(
    "--export-progress",
    action="store_true",
    help="Output intermediate set predictions for DSPN?",
)
parser.add_argument(
    "--full-eval",
    action="store_true",
    help="Use full evaluation set (default: 1/10 of evaluation data)",  # don't need full evaluation when training to save some time
)
parser.add_argument(
    "--mask-feature",
    action="store_true",
    help="Treat mask as a feature to compute loss with",
)
parser.add_argument(
    "--inner-lr",
    type=float,
    default=10000000,
    help="Learning rate of DSPN inner optimisation",
)
parser.add_argument(
    "--iters",
    type=int,
    default=10,
    help="How many DSPN inner optimisation iteration to take",
)
parser.add_argument(
    "--huber-repr",
    type=float,
    default=10,
    help="Scaling of representation loss term for DSPN supervised learning",
)
parser.add_argument(
    "--loss",
    choices=["hungarian", "chamfer"],
    default="hungarian",
    help="Type of loss used",
)

parser.add_argument(
    '--thread', type=int, default=3, help='number of threads'
    )
parser.add_argument(
    '--output', action="store_true", help='if output prediction')
args = parser.parse_args()

#torch.set_num_threads(1)

dataname=args.dataname
vNum = args.vNum

trainNum =args.trainNum
valNum =args.valNum
testNum =args.testNum
pairMax=2500


#simulation times, small number for fast testing
infTimes = 300
#get data
path = os.getcwd() 
data_path=os.path.abspath(os.path.join(path, os.pardir))+"/data"
pairPath = "{}/{}/{}_pair_{}".format(data_path,dataname,dataname,pairMax)
graphPath = "{}/{}/{}_diffusionModel".format(data_path,dataname,dataname)

instance = InputInstance(graphPath, None, 0, vNum, 0, None, 
                 None, None)
    
lineNums=(np.random.permutation(pairMax)*5)
#lineNums.sort()
allLineNums=lineNums[0: trainNum+testNum+valNum]
allLineNums.sort()
trainLineNums=lineNums[0:trainNum]
trainLineNums.sort()
testLineNums=lineNums[trainNum: trainNum+testNum]
testLineNums.sort()
valLineNums=lineNums[trainNum+testNum : trainNum+testNum+valNum]
valLineNums.sort()
    
    
print('Building dataset...')
dataset_all = data.TestDataset(pairPath, trainNum+testNum+valNum, args.batch_size, allLineNums, vNum,train=True, full=args.full_eval)
max_set_size=dataset_all.max_size
dataset_train = data.TestDataset(pairPath, trainNum, args.batch_size, trainLineNums, vNum, train=True, full=args.full_eval,max_size=max_set_size)
dataset_test = data.TestDataset(pairPath,  testNum, args.batch_size, testLineNums, vNum, train=False, full=args.full_eval,max_size=max_set_size)
dataset_val = data.TestDataset(pairPath,  valNum, args.batch_size, valLineNums, vNum, train=False, full=args.full_eval,max_size=max_set_size)
print("max_set_size {}".format(max_set_size))


args.set_size=max_set_size
args.vNum=vNum
net = model.build_net(args)

if not args.no_cuda:
    net = net.cuda()

if args.multi_gpu:
    net = torch.nn.DataParallel(net)


optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
decayRate = 0.9
my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
#optimizer = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=args.lr)
 
def run(net, dataset,  optimizer, train=True, epoch=10, pool=None, valDataset=None):
    if train:
        net.train()    
        print("train")
        torch.set_grad_enabled(True)
    else:
        net.eval()
        torch.set_grad_enabled(False)

    stop=False;
    true_export = []
    pred_export = []
    true_import = []
    iters_per_epoch=int(dataset.size/dataset.batch_size)   
    print("epoch {}".format(epoch))
    min_loss=sys.maxsize
    for i in range(iters_per_epoch):
        print("batch {} {} {} ".format(epoch ,iters_per_epoch, i))

        sample=dataset.data[i*dataset.batch_size : (i+1)*dataset.batch_size]
        
        input=torch.stack([x[0] for x in sample])
        target_set=torch.stack([x[1] for x in sample])
        #target_mask=torch.stack([x[2] for x in sample])
        
        #print()
        (progress, masks, evals, gradn), (y_enc, y_label) = net(
            input, target_set, max_set_size=dataset.max_size)
                    
        
        # Only use representation loss with DSPN and when doing general supervised prediction, not when auto-encoding

        repr_loss = 10 * F.smooth_l1_loss(y_enc, y_label)
        loss = repr_loss
        
                   
        set_true, set_pred = [], []
        for i in range(len(target_set)):
            set_true.append(utils.matrix_to_one_hot(target_set[i].detach().cpu(), vNum))
            set_pred.append(progress[-1][i].detach().cpu())
                            
      
        set_loss = []
        for i in range(len(set_pred)):
            set_pred[i].requires_grad=True
            #set_loss.append(F.smooth_l1_loss(set_pred[i], set_true[i]*1000000000))
            set_loss.append(F.binary_cross_entropy_with_logits(set_pred[i], set_true[i]))
            
        if args.no_cuda:
            set_loss = torch.tensor(set_loss, dtype=torch.float64, requires_grad=True)
        else:
            set_loss = set_loss.cuda()
        
        #loss = set_loss.mean()
        loss = set_loss.mean() + repr_loss.mean()
        print('\n set loss: ', set_loss.mean().item())
        print('repr loss: ', repr_loss.mean().item())
        
        if train:
            #print("optimizer.zero_grad()...")
            optimizer.zero_grad()
            #print("loss.backward()...")
            loss.backward()
            #print("optimizer.step()...")
            optimizer.step()
            #print("optimizer.step() done")
            my_lr_scheduler.step()      

        if train:
            sample_val=valDataset.data
            
            input_val=torch.stack([x[0] for x in sample_val]).detach().cpu()
            target_set_val=torch.stack([x[1] for x in sample_val]).detach().cpu()
          
            
            
            (progress_val, masks, evals, gradn), (y_enc_val, y_label_val) = net(
                input_val, target_set_val, valDataset.max_size)
                        
            
            # Only use representation loss with DSPN and when doing general supervised prediction, not when auto-encoding

            repr_loss_val = 10 * F.smooth_l1_loss(y_enc_val, y_label_val)
            #loss_val = repr_loss_val
            
                       
            set_true_val, set_pred_val = [], []
            for i in range(len(target_set_val)):
                set_true_val.append(utils.matrix_to_one_hot(target_set_val[i].detach().cpu(), vNum))
                set_pred_val.append(progress_val[-1][i].detach().cpu())
                                
          
            set_loss_val = []
            for i in range(len(set_pred_val)):
                set_pred_val[i].requires_grad=True
                #print(set_pred_val[i].shape)
                #print(set_true_val[i].shape)
                set_loss_val.append(F.binary_cross_entropy_with_logits(set_pred_val[i], set_true_val[i]))
                
            if args.no_cuda:
                set_loss_val = torch.tensor(set_loss_val, dtype=torch.float64)
            else:
                set_loss_val = set_loss_val.cuda()
            
            print('\n set val loss: ', set_loss_val.mean().item())
            print('repr  val loss: ', repr_loss_val.mean().item())
            
            
            true_export_val = []
            pred_export_val = []
            true_import_val = []
    
            for p, s, pro in zip(target_set_val, input_val, progress_val[-1]):
                true_export_val.append(utils.one_hot_to_number(p.detach().cpu()))
                true_import_val.append(utils.one_hot_to_number(s.detach().cpu()))
                k=len(utils.one_hot_to_number(p.detach().cpu()))
                pred_export_val.append(torch.topk(pro.cpu().detach(), k=k).indices.numpy())
            #utils.runTesting(true_import_val, true_export_val, pred_export_val, instance)
            
            #print(len(utils.one_hot_to_number(p.detach().cpu())))
            #print(len(torch.topk(pro.cpu().detach(), k=k).indices.numpy()))
            print("*********")
            if set_loss_val.mean().item()>min_loss:
                stop=True;
                break;
            else:
                min_loss=set_loss_val.mean().item()
  
           

        
       
        for p, s, pro in zip(target_set, input, progress[-1]):
            true_export.append(utils.one_hot_to_number(p.detach().cpu()))
            true_import.append(utils.one_hot_to_number(s.detach().cpu()))
            k=len(utils.one_hot_to_number(p.detach().cpu()))
            pred_export.append(torch.topk(pro.cpu().detach(), k=k).indices.numpy())
            #print(len(utils.one_hot_to_number(p.detach().cpu())))
            #print(len(torch.topk(pro.cpu().detach(), k=k).indices.numpy()))

            
        #progress_steps = []
        #for pro  in progress[-1] :
           
                
                
    print("Loop done")
    print(len(pred_export))
    
    
    if not train:
        utils.runTesting(true_import, true_export, pred_export, instance)
        print(dataname)
        print("trainNum:{}, testNum:{}, infTimes:{} ".format(trainNum, testNum,  infTimes))
    #else:
       # my_lr_scheduler.step()
    return stop
    
for epoch in range(args.epochs):
#    tracker.new_epoch()
    if not args.eval_only:
        print("-----------------------------training....")
        stop=run(net, dataset_train, optimizer, train=True, epoch=epoch, pool=None, valDataset=dataset_val)
    if epoch==args.epochs-1 or True:
        print("-----------------------------testing....")
        run(net, dataset_test, optimizer, train=False, epoch=epoch, pool=None)
        #break;
    if args.eval_only:
        break
    


    
    
    
    
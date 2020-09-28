# -*- coding: utf-8 -*-

import scipy.optimize
import torch
import torch.nn.functional as F
import numpy as np
import multiprocessing

def hungarian_loss(predictions, targets, thread_pool):
    # predictions and targets shape :: (n, c, s)
    predictions, targets = outer(predictions, targets)
    # squared_error shape :: (n, s, s)
    squared_error = F.smooth_l1_loss(predictions, targets.expand_as(predictions), reduction="none").mean(1)

    squared_error_np = squared_error.detach().cpu().numpy()
    indices = thread_pool.map(hungarian_loss_per_sample, squared_error_np)
    losses = [
        sample[row_idx, col_idx].mean()
        for sample, (row_idx, col_idx) in zip(squared_error, indices)
    ]
    total_loss = torch.mean(torch.stack(list(losses)))
    return total_loss


def hungarian_loss_per_sample(sample_np):
    return scipy.optimize.linear_sum_assignment(sample_np)


def one_hot_encoder(data, max_value):
    shape = (data.size, max_value+1)
    one_hot = np.zeros(shape)
    rows = np.arange(data.size)
    one_hot[rows, data] = 1
    
    return one_hot


def chamfer_loss(predictions, targets):
    # predictions and targets shape :: (k, n, c, s)
    predictions, targets = outer(predictions, targets)
    # squared_error shape :: (k, n, s, s)
    squared_error = F.smooth_l1_loss(predictions, targets.expand_as(predictions), reduction="none").mean(2)
    loss = squared_error.min(2)[0] + squared_error.min(3)[0]
    return loss.view(loss.size(0), -1).mean(1)

'''
def tensor_to_set(my_tensor):
    my_tensor = my_tensor.numpy()
    set_list = []
    for i in range(len(my_tensor)):
        temp = my_tensor[i].flatten()
        temp = [math.ceil(j) for j in temp]
        set_list.append(set(temp))
    
    return set_list
  '''  

def tensor_to_set(my_tensor):
    my_tensor = my_tensor.numpy()
    set_list = []
    for i in my_tensor:
        temp_list = []
        for j in i:
            temp_list.append(np.argmax(j))
        set_list.append(set(temp_list))
    
    return set_list


def one_hot_to_number(matrix):
    number = []
    #print("-----------")
    #print(matrix.shape)
    for i in torch.transpose(matrix,0,1):
        #print(i.shape)
        number.append(torch.argmax(i).item())
    #print(number)
    #print("-----------")
    return list(set(number))


def matrix_to_one_hot(matrix, target_number):
    #print("matrix_to_one_hot")
    #print(matrix.shape)
    indices = torch.argmax(matrix, dim=0)
    #print(indices)
    #print(indices)
    indices_tensor = torch.zeros(target_number)
    #print("matrix_to_one_hot")
    for i in indices:
        indices_tensor[i] = 1
    
    return indices_tensor
    

def scatter_masked(tensor, mask, binned=False, threshold=None):
    s = tensor[0].detach().cpu()
    mask = mask[0].detach().clamp(min=0, max=1).cpu()
    if binned:
        s = s * 128
        s = s.view(-1, s.size(-1))
        mask = mask.view(-1)
    if threshold is not None:
        keep = mask.view(-1) > threshold
        s = s[:, keep]
        mask = mask[keep]
    return s, mask


def outer(a, b=None):
    """ Compute outer product between a and b (or a and a if b is not specified). """
    if b is None:
        b = a
    size_a = tuple(a.size()) + (b.size()[-1],)
    size_b = tuple(b.size()) + (a.size()[-1],)
    a = a.unsqueeze(dim=-1).expand(*size_a)
    b = b.unsqueeze(dim=-2).expand(*size_b)
    return a, b

def runTesting(X_test, Y_test, Y_pred, instance):
    #print("runTesting")
    X_test_set=[]
    Y_test_set=[]
    Y_pred_set=[]
    for x, y, y_hat in zip (X_test, Y_test, Y_pred):
        X_test_set.append(vec_to_set(x))
        Y_test_set.append(vec_to_set(y))
        Y_pred_set.append(vec_to_set(y_hat))
        '''
        x.sort()
        print(x)
        y.sort()
        print(y)
        y_hat.sort()
        print(y_hat)
        print("--------------------")
        '''
    #testing(X_test_set,Y_test_set,Y_pred_set,instance)
    testingNew(X_test_set,Y_test_set,Y_pred_set,instance)
        
def testing(X_test,Y_test,Y_pred,instance,infTimes=256):
    print("Testing Started")
    reduce_percent_opt=[]
    reduce_percent_pre = []
    com_to_opt = []
    error_abs = []
    error_ratio = []
    for x, y, y_pred in zip(X_test, Y_test,  Y_pred):
        influence_x=instance.testInfluence_0(x,{}, infTimes, 3)
        influence_y=instance.testInfluence_0(x,y, infTimes, 3)
        influence_y_pred=instance.testInfluence_0(x,y_pred, infTimes, 3)
        #print("{} {} {} ".format(influence_x,influence_y,influence_y_pred))
        reduce_percent_opt.append((influence_x-influence_y)/influence_x)
        reduce_percent_pre.append( (influence_x-influence_y_pred)/influence_x)
        com_to_opt.append((influence_x-influence_y_pred)/(influence_x-influence_y+0.01))
        error_abs.append((influence_y_pred-influence_y))
        error_ratio.append((influence_y_pred-influence_y)/influence_y)
        #print()
   
    print("error_abs: {} +- {}".format(np.mean(np.array(error_abs)), np.std(np.array(error_abs))))
    print("com_to_opt: {} +- {}".format(np.mean(np.array(com_to_opt)), np.std(np.array(com_to_opt))))
    
def testingNew(X_test,Y_test,Y_pred,instance,infTimes=270):
    #print("Testing Started")

    thread = 2;
    block_size =int (infTimes/thread);
    p = multiprocessing.Pool(thread)
    
    influence_Xs = p.starmap(instance.testInfluence_0_block, ((X_test[i*block_size:(i+1)*block_size], infTimes) for i in range(thread)),1)
    p.close()
    p.join()
    
    p = multiprocessing.Pool(thread)
    influence_Ys = p.starmap(instance.testInfluence_0_block, ((X_test[i*block_size:(i+1)*block_size], infTimes, Y_test[i*block_size:(i+1)*block_size]) for i in range(thread)),1)
    p.close()
    p.join()
    
    p = multiprocessing.Pool(thread)
    influence_Y_preds = p.starmap(instance.testInfluence_0_block, ((X_test[i*block_size:(i+1)*block_size], infTimes, Y_pred[i*block_size:(i+1)*block_size]) for i in range(thread)),1)
    p.close()
    p.join()
    
    
    influence_X=[]
    influence_Y=[]
    influence_Y_pred=[]
    for i in range(thread):
        influence_X.extend(influence_Xs[i])
        influence_Y.extend(influence_Ys[i])
        influence_Y_pred.extend(influence_Y_preds[i])
    
    
    reduce_percent_opt=[]
    reduce_percent_pre = []
    com_to_opt = []
    error_abs = []
    error_ratio = []
    for influence_x, influence_y, influence_y_pred in zip(influence_X, influence_Y, influence_Y_pred):
        #print("{} {} {} ".format(influence_x,influence_y,influence_y_pred))
        reduce_percent_opt.append((influence_x-influence_y)/influence_x)
        reduce_percent_pre.append( (influence_x-influence_y_pred)/influence_x)
        com_to_opt.append((influence_x-influence_y_pred)/(influence_x-influence_y+0.01))
        error_abs.append((influence_y_pred-influence_y))
        error_ratio.append((influence_y_pred-influence_y)/influence_y)
        #print()

    print("error_abs: {} +- {}".format(np.mean(np.array(error_abs)), np.std(np.array(error_abs))))
    print("error_ratio: {} +- {}".format(np.mean(np.array(error_ratio)), np.std(np.array(error_ratio))))
    print("reduce_percent_opt: {} +- {}".format(np.mean(np.array(reduce_percent_opt)), np.std(np.array(reduce_percent_opt))))
    print("reduce_percent_pre: {} +- {}".format(np.mean(np.array(reduce_percent_pre)), np.std(np.array(reduce_percent_pre))))
    print("com_to_opt: {} +- {}".format(np.mean(np.array(com_to_opt)), np.std(np.array(com_to_opt))))
        
def vec_to_set(X):
    y=set()
    for x in X:
        y.add(str(x))
    return y
    
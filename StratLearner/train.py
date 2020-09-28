"""
==============================
StratLearner Training
==============================
"""

import numpy as np
from one_slack_ssvm import OneSlackSSVM
from stratLearner import (StratLearn, Utils, InputInstance)
import multiprocessing
import argparse
import os
import sys
from datetime import datetime

class Object(object):
    pass



parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataname',  default='kro', 
                    choices=['kro', 'power768', 'ER512'])
parser.add_argument(
    '--vNum', type=int, default=1024, choices=[1024,768,512],
                    help='kro 1024, power768 768, ER512 512')


parser.add_argument(
    '--featureNum', type=int, default=100,
                    help='number of features (random subgraphs) used in StratLearn ')
parser.add_argument(
    '--featureGenMethod', default='uniform_structure0-01', \
        choices=['uniform_structure1-0','uniform_structure0-01', 'uniform_structure0-005','WC_Weibull_structure'], \
            help='the distribution used for generating features, the choices correspond phi_1^1, phi_0.01^1, phi_0.005^1, phi_+^+')

    
parser.add_argument(
    '--trainNum', type=int, default=270, help='number of training data')  
parser.add_argument(
    '--testNum', type=int, default=270, help='number of testing data')   


parser.add_argument(
    '--thread', type=int, default=1, help='number of threads')

parser.add_argument(
    '--output', default=False, action="store_true", help='if output prediction')


parser.add_argument(
    '--pre_train', default=False,action="store_true", help='if store a pre_train model')


args = parser.parse_args()
utils= Utils()


dataname=args.dataname
vNum = args.vNum



trainNum =args.trainNum
testNum =args.testNum
pairMax=2500

thread = args.thread
verbose=3

#parameter used in SVM
C = 0.01
tol=0.001


featureNum = args.featureNum
featureGenMethod = args.featureGenMethod
if featureGenMethod == "uniform_structure1-0":
    maxFeatureNum=1
    featureNum=1
    max_iter=1
else:
    if featureGenMethod == "WC_Weibull_structure":
        maxFeatureNum=800
        max_iter = 30
    else:
        maxFeatureNum=2000
        max_iter = 30





#define the one-hop loss
balance_para=1000;
loss_type = Object()
loss_type.name="area"
loss_type.weight=1
LAI_method = "fastLazy"
effectAreaNum = 1



#simulation times, small number for testing
infTimes = 1024

#get data
path = os.getcwd() 
data_path=os.path.abspath(os.path.join(path, os.pardir))+"/data"
pair_path = "{}/{}/{}_pair_{}".format(data_path,dataname,dataname,pairMax)
graphPath = "{}/{}/{}_diffusionModel".format(data_path,dataname,dataname)
featurePath = "{}/{}/feature/{}_{}/".format(data_path,dataname,featureGenMethod,maxFeatureNum)


X_train, Y_train, _, _, X_test, Y_test, _, _ = utils.getDataTrainTestRandom(pair_path ,trainNum,testNum, pairMax)
print("data fetched")

instance = InputInstance(graphPath, featurePath, featureNum, vNum, effectAreaNum, 
                         balance_para, loss_type, featureRandom = True, maxFeatureNum = maxFeatureNum,
                         thread = thread, LAI_method=LAI_method)



#**************************OneSlackSSVM
model = StratLearn()
model.initialize(X_train, Y_train, instance)

one_slack_svm = OneSlackSSVM(model, verbose=verbose, C=C, tol=tol, n_jobs=thread,
                         max_iter = max_iter)
one_slack_svm.fit(X_train, Y_train, initialize = False)


if args.pre_train:
    featureIndexes=instance.featureIndexes
    path="pre_train/preTrain_"+dataname+"_"+featureGenMethod+"_"+str(featureNum)
    if os.path.exists(path):
        sys.exit(path+" already exists, please remove the existing file")
    with open(path, 'a') as the_file:
        the_file.write(dataname+"\n")
        the_file.write(str(vNum)+"\n")
        the_file.write(featureGenMethod+"\n")
        the_file.write(str(featureNum)+"\n")
        for index, weight in zip(featureIndexes, one_slack_svm.w):
            the_file.write(str(index))
            the_file.write(" ")
            the_file.write(str(weight))
            the_file.write('\n')
   
    

print("Prediction Started")
Y_pred = one_slack_svm.predict(X_test, featureNum)




print("Testing Started")


block_size =int (testNum/thread);
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
    #print("{} {} {} {} {}".format(influence_x,influence_y,influence_y_pred, influence_x_read, influence_y_read))
    reduce_percent_opt.append((influence_x-influence_y)/influence_x)
    reduce_percent_pre.append( (influence_x-influence_y_pred)/influence_x)
    com_to_opt.append((influence_x-influence_y_pred)/(influence_x-influence_y+0.01))
    error_abs.append((influence_y_pred-influence_y))
    error_ratio.append((influence_y_pred-influence_y)/influence_y)

if args.output:
    now = datetime.now()
    with open(now.strftime("%d-%m-%Y %H:%M:%S"), 'a') as the_file:
        for x_test, y_test, y_pred in zip(X_test,Y_test,Y_pred):
            for target in [x_test, y_test, y_pred]:
                line='';
                for a in target:
                    line += a
                    line += ' '
                line += '\n'
                the_file.write(line)
            the_file.write('\n')


print(dataname)
print('StratLearner')
print("error_abs: {} +- {}".format(np.mean(np.array(error_abs)), np.std(np.array(error_abs))))
print("error_ratio: {} +- {}".format(np.mean(np.array(error_ratio)), np.std(np.array(error_ratio))))
print("reduce_percent_opt: {} +- {}".format(np.mean(np.array(reduce_percent_opt)), np.std(np.array(reduce_percent_opt))))
print("reduce_percent_pre: {} +- {}".format(np.mean(np.array(reduce_percent_pre)), np.std(np.array(reduce_percent_pre))))
print("com_to_opt: {} +- {}".format(np.mean(np.array(com_to_opt)), np.std(np.array(com_to_opt))))

#
print("featureNum:{}, featureGenMethod: {}, c:{} balance_para: {}".format(featureNum, featureGenMethod, C,balance_para))
print("trainNum:{}, testNum:{}, infTimes:{} ".format(trainNum, testNum,  infTimes))
print("loss_type:{}, LAI_method:{}, ".format(loss_type.name, LAI_method))

print("===============================================================")


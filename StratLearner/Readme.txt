To run the pre_train model, use

run_PreTrain.py --path Path --testNum, x

Path is the path of the stored per-trained model. The program will randomly select x testing pairs and output the results.

e.g,

Python run_PreTrain.py --path pre_train/preTrain_ER512_uniform_structure0-01_100 --testNum 270


The pre_model records the indexes of the features and their weights given by StratLearner.




The testing report has the following format

ER512
StratLearner
error_abs: 1.8 +- 2.2575882101811597
error_ratio: 0.11799175281835877 +- 0.13421873729428194
reduce_percent_opt: 0.4597198978861392 +- 0.09092133395565785
reduce_percent_pre: 0.4016695609290439 +- 0.08968904313060766
com_to_opt: 0.8776171711299374 +- 0.11978522153155108
featureNum:800, featureGenMethod: uniform_structure0-01, c:0.01 balance_para: 1000
trainNum:0, testNum:270, infTimes:300 
loss_type:area, LAI_method:fastLazy,

The com_to_opt is the performance ratio used in the paper.




The folder "pre_train" includes a few pre_trained models.

pre_dataname_distribution_featureNum

E.g.: preTrain_ER512_uniform_structure0-01_100: the StratLeaner trained with ER512 with 100 features generated from distribution phi_0.01^1.0

The attached models are trained use 270 training pairs. 

For example, according to our report in Table 1, 

preTrain_ER512_uniform_structure0-01_100 should reproduce com_to_opt as 0.661 (+- 2e-2);
preTrain_ER512_uniform_structure0-01_400 should reproduce com_to_opt as 0.853 (+- 6e-3);
preTrain_ER512_uniform_structure0-01_800 should reproduce com_to_opt as 0.873 (+- 1e-2);
preTrain_ER512_uniform_structure0-01_1600 should reproduce com_to_opt as 0.892 (+- 3e-3).



Thank you.

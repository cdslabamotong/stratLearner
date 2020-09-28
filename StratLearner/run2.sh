python3 train.py --dataname power768 --vNum 768 --featureNum 400 --featureGenMethod uniform_structure0-01 --trainNum 270 --thread 90 --pre_train| tee -a log2.txt
python3 train.py --dataname power768 --vNum 768 --featureNum 400 --featureGenMethod WC_Weibull_structure --trainNum 270 --thread 90 --pre_train| tee -a log2.txt
python3 train.py --dataname ER512 --vNum 512 --featureNum 400 --featureGenMethod uniform_structure0-01 --trainNum 270 --thread 90 --pre_train| tee -a log2.txt
python3 train.py --dataname ER512 --vNum 512 --featureNum 400 --featureGenMethod WC_Weibull_structure --trainNum 270 --thread 90 --pre_train| tee -a log2.txt









########### resnet50
###########          W4A4
python main_D.py --conf_path ./imagenet_resnet50.hocon --id 1 --CE_WEIGHT 1 --BNS_WEIGHT 0.1  --FEATURE_WEIGHT 1   --warmup_epochs 50 --visible_devices 0 --seed 100 --selenet resnet50
###########     GDFQ     W4A4
python main_D.py --conf_path ./imagenet_resnet50.hocon --id 2 --CE_WEIGHT 1 --BNS_WEIGHT 0.1  --FEATURE_WEIGHT 1   --warmup_epochs 50 --visible_devices 1 --seed 100 --selenet resnet50

###########          W6A6
python main_D.py --conf_path ./imagenet_resnet50.hocon --id 1 --CE_WEIGHT 1 --BNS_WEIGHT 0.1  --FEATURE_WEIGHT 1   --warmup_epochs 50 --visible_devices 2 --seed 100 --selenet resnet50
###########     GDFQ     W6A6
python main_D.py --conf_path ./imagenet_resnet50.hocon --id 2 --CE_WEIGHT 1 --BNS_WEIGHT 0.1  --FEATURE_WEIGHT 1   --warmup_epochs 50 --visible_devices 3 --seed 100 --selenet resnet50

###########          W8A8
python main_D.py --conf_path ./imagenet_resnet50.hocon --id 1 --CE_WEIGHT 1 --BNS_WEIGHT 0.1  --FEATURE_WEIGHT 1   --warmup_epochs 50 --visible_devices 4 --seed 100 --selenet resnet50
###########     GDFQ     W8A8
python main_D.py --conf_path ./imagenet_resnet50.hocon --id 2 --CE_WEIGHT 1 --BNS_WEIGHT 0.1  --FEATURE_WEIGHT 1   --warmup_epochs 50 --visible_devices 5 --seed 100 --selenet resnet50


python main_D.py --conf_path ./imagenet_resnet18.hocon --id 7 --CE_WEIGHT 1 --BNS_WEIGHT 10  --FEATURE_WEIGHT 1   --warmup_epochs 50 --visible_devices 4 --seed 100  --selenet resnet18
python main_D.py --conf_path ./imagenet_resnet18.hocon --id 8 --CE_WEIGHT 1 --BNS_WEIGHT 0.1  --FEATURE_WEIGHT 1   --warmup_epochs 50 --visible_devices 5 --seed 100  --selenet resnet18
python main_D.py --conf_path ./imagenet_resnet18.hocon --id 9 --CE_WEIGHT 1 --BNS_WEIGHT 1  --FEATURE_WEIGHT 1   --warmup_epochs 50 --visible_devices 6 --seed 100  --selenet resnet18
python main_D.py --conf_path ./imagenet_resnet18.hocon --id 10 --CE_WEIGHT 1 --BNS_WEIGHT 10  --FEATURE_WEIGHT 1   --warmup_epochs 50 --visible_devices 7 --seed 100  --selenet resnet18


###     ShuffleNet ablation study
###########          W4A4
python main_D.py --conf_path ./imagenet_shufflenet.hocon --id 1 --CE_WEIGHT 1 --BNS_WEIGHT 1  --FEATURE_WEIGHT 1   --warmup_epochs 50 --visible_devices 5 --seed 100 --selenet shufflenet


###     resnet18 ablation study
###########          W6A6
python main_D.py --conf_path ./imagenet_resnet18.hocon --id 1 --CE_WEIGHT 1 --BNS_WEIGHT 1  --FEATURE_WEIGHT 1   --warmup_epochs 50 --visible_devices 6 --seed 100  --selenet resnet18
###########          W8A8
python main_D.py --conf_path ./imagenet_resnet18.hocon --id 1 --CE_WEIGHT 1 --BNS_WEIGHT 1  --FEATURE_WEIGHT 1   --warmup_epochs 50 --visible_devices 7 --seed 100  --selenet resnet18

###     resnet50 ablation study
###########          W4A4
python main_D.py --conf_path ./imagenet_resnet50.hocon --id 3 --CE_WEIGHT 1 --BNS_WEIGHT 1  --FEATURE_WEIGHT 1   --warmup_epochs 50 --visible_devices 0 --seed 100  --selenet resnet50
###########          W6A6
python main_D.py --conf_path ./imagenet_resnet50.hocon --id 3 --CE_WEIGHT 1 --BNS_WEIGHT 1  --FEATURE_WEIGHT 1   --warmup_epochs 50 --visible_devices 1 --seed 100  --selenet resnet50
###########          W8A8
python main_D.py --conf_path ./imagenet_resnet50.hocon --id 3 --CE_WEIGHT 1 --BNS_WEIGHT 1  --FEATURE_WEIGHT 1   --warmup_epochs 50 --visible_devices 2 --seed 100  --selenet resnet50




###################################################################################################
######################################  W5A5    ###################################################
###################################################################################################
########### Mobile_v2
###########          W5A5
python main_D.py --conf_path ./imagenet_mobilev2.hocon --id 1 --CE_WEIGHT 1 --BNS_WEIGHT 1  --FEATURE_WEIGHT 1   --warmup_epochs 50 --visible_devices 1 --seed 100 --selenet MobileV2
###########     GDFQ     W5A5
python main_D.py --conf_path ./imagenet_mobilev2.hocon --id 2 --CE_WEIGHT 1 --BNS_WEIGHT 0.1  --FEATURE_WEIGHT 1   --warmup_epochs 50 --visible_devices 2 --seed 100 --selenet MobileV2

########### ShuffleNet
###########          W5A5
python main_D.py --conf_path ./imagenet_shufflenet.hocon --id 1 --CE_WEIGHT 1 --BNS_WEIGHT 1  --FEATURE_WEIGHT 1   --warmup_epochs 50 --visible_devices 3 --seed 100 --selenet shufflenet
###########     GDFQ     W5A5
python main_D.py --conf_path ./imagenet_shufflenet.hocon --id 2 --CE_WEIGHT 1 --BNS_WEIGHT 0.1  --FEATURE_WEIGHT 1   --warmup_epochs 50 --visible_devices 4 --seed 100 --selenet shufflenet

########### resnet50
###########          W5A5
python main_D.py --conf_path ./imagenet_resnet50.hocon --id 1 --CE_WEIGHT 1 --BNS_WEIGHT 1  --FEATURE_WEIGHT 1   --warmup_epochs 50 --visible_devices 5 --seed 100 --selenet resnet50
###########     GDFQ     W5A5
python main_D.py --conf_path ./imagenet_resnet50.hocon --id 2 --CE_WEIGHT 1 --BNS_WEIGHT 0.1  --FEATURE_WEIGHT 1   --warmup_epochs 50 --visible_devices 6 --seed 100 --selenet resnet50


##########      20220322 TEST       #############
python main_D.py --conf_path ./cifar10_resnet20.hocon --id 1000 --CE_WEIGHT 1 --BNS_WEIGHT 1  --FEATURE_WEIGHT 1  --warmup_epochs 20 --visible_devices 3 --seed 100

python main_D.py --conf_path ./imagenet_inceptionv3.hocon --id 1000 --CE_WEIGHT 1 --BNS_WEIGHT 1  --FEATURE_WEIGHT 1  --warmup_epochs 50 --visible_devices 3 --seed 100 --selenet  inceptionv3

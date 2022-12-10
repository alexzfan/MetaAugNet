#!/bin/bash

# EXPERIMENTS TO RUN 
# #Eval 

# To run: 
# bash run.sh

echo TEST RANDOM INITIALIZATIONS 
python maml_higher.py --outer_lr 1e-3 --num_augs 3 --num_inner_steps 1 --aug_net_size 3 --l2_wd 1e-4 --dataset imagenet --num_train_iterations 5000 --train_aug_type learned --aug_noise_prob 0.1 --identity_init_off

echo TEST NOISE EFFECT
python maml_higher.py --outer_lr 1e-3 --num_augs 3 --num_inner_steps 1 --aug_net_size 3 --l2_wd 1e-4 --dataset imagenet --num_train_iterations 5000 --train_aug_type learned --aug_noise_prob 0
python maml_higher.py --outer_lr 1e-3 --num_augs 3 --num_inner_steps 1 --aug_net_size 3 --l2_wd 1e-4 --dataset imagenet --num_train_iterations 5000 --train_aug_type learned --aug_noise_prob 0.1
python maml_higher.py --outer_lr 1e-3 --num_augs 3 --num_inner_steps 1 --aug_net_size 3 --l2_wd 1e-4 --dataset imagenet --num_train_iterations 5000 --train_aug_type learned --aug_noise_prob 0.5
python maml_higher.py --outer_lr 1e-3 --num_augs 3 --num_inner_steps 1 --aug_net_size 3 --l2_wd 1e-4 --dataset imagenet --num_train_iterations 5000 --train_aug_type learned --aug_noise_prob 1

echo TEST num_augs EFFECT WITH NO NOISE 
python maml_higher.py --outer_lr 1e-3 --num_augs 1 --num_inner_steps 1 --aug_net_size 3 --l2_wd 1e-4 --dataset imagenet --num_train_iterations 5000 --train_aug_type learned --aug_noise_prob 0

# echo TEST PRETRAINED 
# python maml_higher.py --outer_lr 1e-3 --num_augs 3 --num_inner_steps 1 --aug_net_size 3 --l2_wd 1e-4 --dataset imagenet --num_train_iterations 5000 --train_aug_type learned --aug_noise_prob 0.1 --pretrained
 

echo TEST BASELINES METHODS 
python maml_higher.py --outer_lr 1e-3 --num_augs 3 --num_inner_steps 1 --aug_net_size 3 --l2_wd 1e-4 --dataset imagenet --num_train_iterations 5000 --aug_noise_prob 0.1 --train_aug_type learned
python maml_higher.py --outer_lr 1e-3 --num_augs 3 --num_inner_steps 1 --aug_net_size 1 --l2_wd 1e-4 --dataset imagenet --num_train_iterations 5000 --aug_noise_prob 0.1 --train_aug_type identity
python maml_higher.py --outer_lr 1e-3 --num_augs 3 --num_inner_steps 1 --aug_net_size 1 --l2_wd 1e-4 --dataset imagenet --num_train_iterations 5000 --aug_noise_prob 0.1 --train_aug_type random_crop_flip
python maml_higher.py --outer_lr 1e-3 --num_augs 3 --num_inner_steps 1 --aug_net_size 1 --l2_wd 1e-4 --dataset imagenet --num_train_iterations 5000 --aug_noise_prob 0.1 --train_aug_type AutoAugment


echo TEST HIGH NUM AUGS
python maml_higher.py --outer_lr 1e-3 --num_augs 10 --num_inner_steps 1 --aug_net_size 3 --l2_wd 1e-4 --dataset imagenet --num_train_iterations 5000 --aug_noise_prob 0.1 --train_aug_type learned


echo TEST AUGMENTATION NETWORK SIZE 
python maml_higher.py --outer_lr 1e-3 --num_augs 3 --num_inner_steps 1 --aug_net_size 1 --l2_wd 1e-4 --dataset imagenet --num_train_iterations 5000 --train_aug_type learned --aug_noise_prob 0.1
python maml_higher.py --outer_lr 1e-3 --num_augs 3 --num_inner_steps 1 --aug_net_size 5 --l2_wd 1e-4 --dataset imagenet --num_train_iterations 5000 --train_aug_type learned --aug_noise_prob 0.1
python maml_higher.py --outer_lr 1e-3 --num_augs 3 --num_inner_steps 1 --aug_net_size 10 --l2_wd 1e-4 --dataset imagenet --num_train_iterations 5000 --train_aug_type learned --aug_noise_prob 0.1



#!/bin/bash

# EXPERIMENTS TO RUN 

# To run: 
# bash run.sh

echo TEST RANDOM INITIALIZATIONS 
python maml_higher.py --outer_lr 1e-3 --num_augs 3 --num_inner_steps 1 --aug_net_size 3 --l2_wd 1e-4 --dataset imagenet --num_train_iterations 5000 --train_aug_type learned --aug_noise_prob 0.1 --identity_init_off

echo TEST NOISE EFFECT
for aug_noise_prob in 0 0.1 0.5 1 
do
    python maml_higher.py --outer_lr 1e-3 --num_augs 3 --num_inner_steps 1 --aug_net_size 3 --l2_wd 1e-4 --dataset imagenet --num_train_iterations 5000 --train_aug_type learned --aug_noise_prob $aug_noise_prob
done

echo TEST num_augs EFFECT WITH NO NOISE 
python maml_higher.py --outer_lr 1e-3 --num_augs 1 --num_inner_steps 1 --aug_net_size 3 --l2_wd 1e-4 --dataset imagenet --num_train_iterations 5000 --train_aug_type learned --aug_noise_prob 0

echo TEST PRETRAINED 
python maml_higher.py --outer_lr 1e-3 --num_augs 3 --num_inner_steps 1 --aug_net_size 3 --l2_wd 1e-4 --dataset imagenet --num_train_iterations 5000 --train_aug_type learned --aug_noise_prob 0.1 --pretrained
 

echo TEST BASELINES METHODS 
for num_augs in 3 10 20
do
    python maml_higher.py --outer_lr 1e-3 --num_augs $num_augs --num_inner_steps 1 --aug_net_size 3 --l2_wd 1e-4 --dataset imagenet --num_train_iterations 5000 --aug_noise_prob 0.1 --train_aug_type learned
    python maml_higher.py --outer_lr 1e-3 --num_augs $num_augs --num_inner_steps 1 --aug_net_size 1 --l2_wd 1e-4 --dataset imagenet --num_train_iterations 5000 --aug_noise_prob 0.1 --train_aug_type identity
    python maml_higher.py --outer_lr 1e-3 --num_augs $num_augs --num_inner_steps 1 --aug_net_size 1 --l2_wd 1e-4 --dataset imagenet --num_train_iterations 5000 --aug_noise_prob 0.1 --train_aug_type random_crop_flip
    python maml_higher.py --outer_lr 1e-3 --num_augs $num_augs --num_inner_steps 1 --aug_net_size 1 --l2_wd 1e-4 --dataset imagenet --num_train_iterations 5000 --aug_noise_prob 0.1 --train_aug_type AutoAugment
done




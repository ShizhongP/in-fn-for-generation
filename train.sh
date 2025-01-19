#!/bin/bash
############
# fine tune#
############

# python train.py \
#     --input_file data/xsum_train_30000.pickle \
#     --eval_input_file data/xsum_validation_3000.pickle \
#     --model_name_or_path t5-small \
#     --output_dir output \
#     --is_train

######################
# leave-one-out train#
######################
for i in {0..0}; do
    echo -e "\033[32m leave one out train on sample_$i \033[0m "
    python train.py \
        --idx $i \
        --input_file data/xsum_train_30000.pickle \
        --eval_input_file data/xsum_validation_3000.pickle \
        --model_name_or_path t5-small \
        --output_dir output \
        --is_train \
        --loo_train \
        --loo_most \
        --resume_from_checkpoint "output/loo_most_0/checkpoint-4500"
    # python train.py \
    #     --idx $i \
    #     --input_file data/xsum_train_30000.pickle \
    #     --eval_input_file data/xsum_validation_3000.pickle \
    #     --model_name_or_path t5-small \
    #     --output_dir output \
    #     --is_train \
    #     --loo_train \
    #     --loo_least
done

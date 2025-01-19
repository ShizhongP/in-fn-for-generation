#!/bin/bash
sample_data_path="data/xsum-sample_10.pickle"
# for i in {0..9}; do
#     echo -e "\033[32m leave one out eval on sample_$i \033[0m "
#     python eval.py \
#         --eval_data_idx $i \
#         --eval_data_path $sample_data_path \
#         --eval_all
#         # --eval \
#         # --eval_loo_most \
#         # --eval_loo_least 
# done
python eval.py \
    --eval_data_path $sample_data_path \
    --eval_all

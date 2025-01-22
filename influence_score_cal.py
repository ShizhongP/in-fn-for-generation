from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


import torch
import bert_util
import torch.autograd as autograd
from tqdm import tqdm
import pickle
import os
import sys
import bert_util
import numpy as np


class args:

    ## common args
    seed = 42
    model_name = "t5-base"
    model_cache_dir = "./model"

    ## influence args
    output_dir = "output"
    loss_scale = 1e-2
    influence_on_decision = True
    damping = 3e-3
    lissa_depth = 0.15
    lissa_repeat = 1


def main():

    model = AutoModelForSeq2SeqLM.from_pretrained("t5-base", cache_dir="./model")

    param_optimizer = list(model.named_parameters())
    for n, p in param_optimizer:
        print(n)

    frozen = ["shared.weight"]

    param_influence = []
    for n, p in param_optimizer:
        if not any(fr in n for fr in frozen):
            param_influence.append(p)
        elif "shared.weight" in n:
            pass  # need gradients through embedding layer for computing saliency map
        else:
            p.requires_grad = False

    param_shape_tensor = []
    param_size = 0
    for p in param_influence:
        tmp_p = p.clone().detach()
        param_shape_tensor.append(tmp_p)
        param_size += torch.numel(tmp_p)

    print("  Parameter size = %d" % param_size)

    tokenized_datasets = pickle.load(open("data/xsum_train_30000.pickle", "rb"))
    influence_fn_examples = pickle.load(open("data/xsum-sample_10.pickle", "rb"))

    def to_tensor(x):
        return torch.tensor(x, dtype=torch.long).unsqueeze(0)

    input_ids = [
        torch.tensor(example["input_ids"], dtype=torch.long).unsqueeze(0)
        for example in influence_fn_examples
    ]
    label_ids = [
        torch.tensor(example["labels"], dtype=torch.long).unsqueeze(0)
        for example in influence_fn_examples
    ]
    attn_mask = [
        torch.tensor(example["attention_mask"], dtype=torch.long).unsqueeze(0)
        for example in influence_fn_examples
    ]

    # for each test sample, we should compute the influence socre on train dataset
    for idx, (input_id, label_id, attention_mask) in enumerate(
        zip(input_ids, label_ids, attn_mask)
    ):
        print(
            f"====================test example: {idx}======================================"
        )
        input_id = input_id.to(model.device)
        label_id = label_id.to(model.device)
        attention_mask = attention_mask.to(model.device)

        # get test example grad
        model.zero_grad()
        output = model(input_id, attention_mask=attention_mask, labels=label_id)
        # test_loss = output.loss
        # *scaled the loss to avoid the grad is too large
        scaled_loss = output.loss * args.loss_scale
        print(" loss:", scaled_loss)

        # test_grads = autograd.grad(test_loss, param_influence)
        test_grads = autograd.grad(scaled_loss, param_influence)

        # reload train dataset
        train_dataloader_lissa = tokenized_datasets
        print("len of traindataset", len(train_dataloader_lissa))

        device = model.device

        ######## IHVP ########
        model.train()
        print("######## START COMPUTING IHVP ########")
        inverse_hvp = bert_util.get_inverse_hvp_lissa(
            test_grads,
            model,
            device,
            param_influence,
            train_dataloader_lissa,
            loss_scale=args.loss_scale,
            damping=args.damping,
            num_samples=args.lissa_repeat,
            recursion_depth=int(len(train_dataloader_lissa) * args.lissa_depth),
        )
        print("######## FINISHED COMPUTING IHVP ########")
        print("inverse_hvp:", inverse_hvp)

        influences = np.zeros(len(train_dataloader_lissa))
        train_tok_sal_lists = []
        for train_idx, sample in enumerate(
            tqdm(train_dataloader_lissa, desc="Train set index")
        ):
            print(train_idx)
            (doc, summ, _, _input_ids, _input_mask, _label_ids) = sample.values()

            _input_ids = to_tensor(_input_ids).to(device)
            _input_mask = to_tensor(_input_mask).to(device)
            _label_ids = to_tensor(_label_ids).to(device)

            ######## L_TRAIN GRADIENT ########
            model.zero_grad()
            output = model(_input_ids, attention_mask=_input_mask, labels=_label_ids)
            output_len = output.logits.size(1)
            train_loss = output.loss / output_len
            train_grads = autograd.grad(train_loss, param_influence)
            influences[train_idx] = torch.dot(
                inverse_hvp, bert_util.gather_flat_grad(train_grads)
            ).item()
            # print(influences[train_idx])
        if args.influence_on_decision:
            pickle.dump(
                influences,
                open(
                    os.path.join(
                        args.output_dir, "influences_test_" + str(idx) + ".pkl"
                    ),
                    "wb",
                ),
            )
        else:
            pickle.dump(
                influences,
                open(
                    os.path.join(
                        args.output_dir, "influences_on_x_test_" + str(idx) + ".pkl"
                    ),
                    "wb",
                ),
            )


if __name__ == "__main__":

    main()

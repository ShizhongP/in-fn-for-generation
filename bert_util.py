from __future__ import absolute_import, division, print_function


import random

import numpy as np
import torch


import torch.autograd as autograd
from scipy import stats


################ functions for influence function ################


def gather_flat_grad(grads):
    views = []
    for p in grads:
        if p.data.is_sparse:
            view = p.data.to_dense().view(-1)
        else:
            view = p.data.view(-1)
        views.append(view)
    return torch.cat(views, 0)


def unflatten_to_param_dim(x, param_shape_tensor):
    tar_p = []
    ptr = 0
    for p in param_shape_tensor:
        len_p = torch.numel(p)
        tmp = x[ptr : ptr + len_p].view(p.shape)
        tar_p.append(tmp)
        ptr += len_p
    return tar_p


def hv(loss, model_params, v):  # according to pytorch issue #24004
    #     s = time.time()
    # calculate the grad of train loss on model_params

    grad = autograd.grad(
        outputs=loss, inputs=model_params, create_graph=True, retain_graph=True
    )
    #     e1 = time.time()
    # check NaN
    # if any(torch.isnan(g).any() for g in grad):
    #     print("Gradient contains NaN")
    #     return None
    # calculate the grad of model_params(iterated test_grad)
    # !ERROR: Hv has NaN
    Hv = autograd.grad(outputs=grad, inputs=model_params, grad_outputs=v)
    #     e2 = time.time()
    #     print('1st back prop: {} sec. 2nd back prop: {} sec'.format(e1-s, e2-e1))

    # if any(torch.isnan(h).any() for h in Hv):
    #     print("Hessian-vector product contains NaN")
    #     return None
    return Hv


######## LiSSA ########


def to_tensor(x):
    return torch.tensor(x, dtype=torch.long).unsqueeze(0)


def get_inverse_hvp_lissa(
    v,  # test grad
    model,  # model
    device,
    param_influence,  # model parameters list to calculate influence
    train_loader,
    damping,  # damping parameters to stabilize the inverse HVP calculation
    num_samples,
    recursion_depth,
    loss_scale=1,
    scale=1e4,
):
    ihvp = None
    for i in range(num_samples):
        cur_estimate = v
        lissa_data_iterator = iter(train_loader)
        for j in range(recursion_depth):
            try:
                doc, summ, _, input_ids, input_mask, label_id = next(
                    lissa_data_iterator
                ).values()
            except StopIteration:
                lissa_data_iterator = iter(train_loader)
                doc, summ, _, input_ids, input_mask, label_id = next(
                    lissa_data_iterator
                ).values()

            input_ids = to_tensor(input_ids).to(device)
            input_mask = to_tensor(input_mask).to(device)
            label_id = to_tensor(label_id).to(device)

            model.zero_grad()
            output = model(input_ids, attention_mask=input_mask, labels=label_id)
            output_len = output.logits.size(1)
            train_loss = output.loss / output_len

            # !ERROR: hvp has NaN values
            hvp = hv(train_loss, param_influence, cur_estimate)

            cur_estimate = [
                _a + (1 - damping) * _b - _c / scale
                for _a, _b, _c in zip(v, cur_estimate, hvp)
            ]
            if (j % 200 == 0) or (j == recursion_depth - 1):

                print(
                    "Recursion at depth %s: norm is %f"
                    % (
                        j,
                        np.linalg.norm(
                            gather_flat_grad(cur_estimate)
                            .to(torch.float32)
                            .cpu()
                            .numpy()
                        ),
                    )
                )

        if ihvp == None:
            ihvp = [_a / scale for _a in cur_estimate]
        else:
            ihvp = [_a + _b / scale for _a, _b in zip(ihvp, cur_estimate)]
    return_ihvp = gather_flat_grad(ihvp)
    return_ihvp /= num_samples
    return return_ihvp


################


# adapted from AllenNLP Interpret
def _register_embedding_list_hook(model, embeddings_list, model_type):
    def forward_hook(module, inputs, output):
        embeddings_list.append(output.squeeze(0).clone().cpu().detach().numpy())

    if model_type == "BERT":
        embedding_layer = model.bert.embeddings.word_embeddings
    elif model_type == "LSTM":
        embedding_layer = model.my_word_embeddings
    else:
        raise ValueError("Current model type not supported.")
    handle = embedding_layer.register_forward_hook(forward_hook)
    return handle


def _register_embedding_gradient_hooks(model, embeddings_gradients, model_type):
    def hook_layers(module, grad_in, grad_out):
        embeddings_gradients.append(grad_out[0])

    if model_type == "BERT":
        embedding_layer = model.bert.embeddings.word_embeddings
    elif model_type == "LSTM":
        embedding_layer = model.my_word_embeddings
    else:
        raise ValueError("Current model type not supported.")
    hook = embedding_layer.register_backward_hook(hook_layers)
    return hook


def saliency_map(
    model, input_ids, segment_ids, input_mask, pred_label_ids, model_type="BERT"
):
    embeddings_list = []
    handle = _register_embedding_list_hook(model, embeddings_list, model_type)
    embeddings_gradients = []
    hook = _register_embedding_gradient_hooks(model, embeddings_gradients, model_type)

    model.zero_grad()
    _loss = model(input_ids, segment_ids, input_mask, pred_label_ids)
    _loss.backward()
    handle.remove()
    hook.remove()

    saliency_grad = embeddings_gradients[0].detach().cpu().numpy()
    saliency_grad = np.sum(saliency_grad[0] * embeddings_list[0], axis=1)
    norm = np.linalg.norm(saliency_grad, ord=1)
    #     saliency_grad = [math.fabs(e) / norm for e in saliency_grad]
    saliency_grad = [
        (-e) / norm for e in saliency_grad
    ]  # negative gradient for loss means positive influence on decision
    return saliency_grad


################


def get_diff_input_masks(input_mask, test_tok_sal_list):
    sal_scores = np.array([sal for tok, sal in test_tok_sal_list])
    sal_ordered_ix = np.argsort(sal_scores)
    invalid_ix = []
    for i, (tok, sal) in enumerate(test_tok_sal_list):
        if (
            tok == "[CLS]" or tok == "[SEP]" or "##" in tok
        ):  # would not mask [CLS] or [SEP]
            invalid_ix.append(i)
    cleaned_sal_ordered_ix = []
    for sal_ix in sal_ordered_ix:
        if sal_ix in invalid_ix:
            continue
        else:
            cleaned_sal_ordered_ix.append(sal_ix)

    # add zero and random
    abs_sal_ordered_ix = np.argsort(np.absolute(sal_scores))
    cleaned_abs_sal_ordered_ix = []
    for sal_ix in abs_sal_ordered_ix:
        if sal_ix in invalid_ix:
            continue
        else:
            cleaned_abs_sal_ordered_ix.append(sal_ix)

    #     mask_ix = (cleaned_sal_ordered_ix[0], cleaned_sal_ordered_ix[int(len(cleaned_sal_ordered_ix)/2)], cleaned_sal_ordered_ix[-1])
    mask_ix = (
        cleaned_sal_ordered_ix[0],
        cleaned_sal_ordered_ix[int(len(cleaned_sal_ordered_ix) / 2)],
        cleaned_sal_ordered_ix[-1],
        cleaned_abs_sal_ordered_ix[0],
        random.choice(cleaned_sal_ordered_ix),
    )  # lowest, median, highest, zero, random
    diff_input_masks = []
    for mi in mask_ix:
        diff_input_mask = input_mask.clone()
        diff_input_mask[0][mi] = 0
        diff_input_masks.append(diff_input_mask)
    return diff_input_masks, mask_ix


def influence_distance(orig_influences, alt_influences, top_percentage=0.01):
    orig_influences = stats.zscore(orig_influences)
    alt_influences = stats.zscore(alt_influences)
    orig_sorted_ix = list(np.argsort(orig_influences))
    orig_sorted_ix.reverse()
    alt_sorted_ix = list(np.argsort(alt_influences))
    alt_sorted_ix.reverse()
    num_top = int(len(orig_influences) * top_percentage)

    orig_top_ix = orig_sorted_ix[:num_top]
    alt_top_ix = alt_sorted_ix[:num_top]
    orig_top_ix_set = set(orig_top_ix)
    alt_top_ix_set = set(alt_top_ix)
    ix_intersection = list(orig_top_ix_set.intersection(alt_top_ix_set))

    return len(ix_intersection) / num_top

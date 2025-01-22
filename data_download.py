# from huggingface_hub import snapshot_download
from transformers import AutoModelForSeq2SeqLM
import random
import torch
import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    AutoTokenizer,
)


class args:

    ## common args
    seed = 42
    model_name = "t5-base"
    model_cache_dir = "./model"

    is_limit_num_of_tran_and_eval_samples = (
        True  # if True, You should setting num_train_examples and num_evaluate_examples
    )
    ## training args
    is_train = True
    batch_size = 4
    num_train_examples = 30000
    epoch = 1

    max_input_length = 1024
    max_target_length = 128
    # output_dir = "./result/t5-small-test-summarization"

    # evaluate args
    is_eval = True
    num_evaluate_examples = 3000
    # check_point = f"result/t5-small-test-summarization/checkpoint-51012"
    check_point = f"./result/t5-small-test-summarization-{num_train_examples}/checkpoint-{(num_train_examples+batch_size-1)//batch_size*epoch}"

    ## influence args
    loss_scale = 1e-2
    influence_on_decision = True
    damping = 3e-3
    lissa_depth = 0.15
    lissa_repeat = 1


if __name__ == "__main__":

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, cache_dir="./model")

    from datasets import load_dataset

    raw_datasets = load_dataset("EdinburghNLP/xsum", cache_dir="./data")

    raw_datasets["train"] = (
        raw_datasets["train"]
        .shuffle(seed=args.seed)
        .select(range(args.num_train_examples))
    )
    raw_datasets["validation"] = (
        raw_datasets["validation"]
        .shuffle(args.seed)
        .select(range(args.num_evaluate_examples))
    )
    raw_datasets["test"] = (
        raw_datasets["test"]
        .shuffle(args.seed)
        .select(range(args.num_evaluate_examples))
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir="./model")
    prefix = "summary:"

    max_input_length = args.max_input_length
    max_target_length = args.max_target_length

    def preprocess_function(examples):

        inputs = [prefix + doc for doc in examples["document"]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["summary"], max_length=max_target_length, truncation=True
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

    import pickle

    with open("data/xsum_train_30000.pickle", "wb") as f:
        pickle.dump(tokenized_datasets["train"], f)
    with open("data/xsum_validation_3000.pickle", "wb") as f:
        pickle.dump(tokenized_datasets["validation"], f)
    with open("data/xsum_test_3000.pickle", "wb") as f:
        pickle.dump(tokenized_datasets["test"], f)

    # samples for influence-func test
    influence_fn_examples = (
        tokenized_datasets["test"].shuffle(args.seed).select(range(10))
    )
    save_path = "./data/xsum-sample_10.pickle"
    with open(save_path, "wb") as f:
        pickle.dump(influence_fn_examples, f)

    import nltk

    nltk.download("punkt_tab")

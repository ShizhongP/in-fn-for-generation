from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
import argparse
import pickle
import torch
import numpy as np
import random
import nltk
import numpy as np
from datasets import load_metric
from rouge_score import rouge_scorer
import nltk
import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = [
        "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds
    ]
    decoded_labels = [
        "\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels
    ]

    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    # Add mean generated length
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
    ]
    result["gen_len"] = np.mean(prediction_lens)
    return {k: round(v, 4) for k, v in result.items()}


def train(model, dataset, output_dir):
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    train_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=args.epoch,
        predict_with_generate=True,
        fp16=True,
        disable_tqdm=False,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    if args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()


def main():

    if not args.loo_train:
        output_dir = args.output_dir + "/finetune"
        train(model, dataset, output_dir)

    if args.loo_train:
        sample_idx = args.idx
        influences = pickle.load(
            open(
                f"result/t5-small-test-summarization-30000/influences_test_{sample_idx}.pkl",
                "rb",
            ),
        )
        influential_idx = np.argsort(influences)  # sort from smallest to largest
        exclude_most_influential_idx = influential_idx[:-3000]
        exclude_least_influential_idx = influential_idx[3000:]

        exclude_most_influential_dataset = dataset.select(exclude_most_influential_idx)
        exclude_least_influential_dataset = dataset.select(
            exclude_least_influential_idx
        )

        exclude_most_influential_output_dir = (
            args.output_dir + "/" + f"loo_most_{sample_idx}"
        )
        exclude_least_influential_output_dir = (
            args.output_dir + "/" + f"loo_least_{sample_idx}"
        )

        if args.is_train:
            # train exclude most influential data
            if args.loo_most:
                print("Traing excluded most influential data")
                train(
                    model,
                    exclude_most_influential_dataset,
                    output_dir=exclude_most_influential_output_dir,
                )
            # train exclude least influential data
            if args.loo_least:
                print("Traing excluded least influential data")
                train(
                    model,
                    exclude_least_influential_dataset,
                    output_dir=exclude_least_influential_output_dir,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leave one out training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--idx",
        type=int,
        default=0,
        required=False,
        help="Index of the sample for Leave-one-out-train",
    )
    parser.add_argument("--model_name_or_path", type=str, help="Model name")
    parser.add_argument("--input_file", type=str, help="Input file")
    parser.add_argument("--eval_input_file", type=str, help="Evaluation input file")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--epoch", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
    parser.add_argument(
        "--resume_from_training", default=None, type=str, help="Resume from training"
    )
    parser.add_argument(
        "--loo_train", action="store_true", help="Leave-one-out training"
    )
    parser.add_argument("--loo_most", action="store_true", help="Leave-one-out most")
    parser.add_argument("--loo_least", action="store_true", help="Leave-one-out least")
    parser.add_argument("--is_train", action="store_true", help="Whether to train")
    parser.add_argument("--is_eval", action="store_true", help="Whether to Evaluate")
    parser.add_argument("--resume_from_checkpoint", default=None, type=str)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, cache_dir="model"
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path, cache_dir="model"
    )
    dataset = pickle.load(open(args.input_file, "rb"))
    eval_dataset = pickle.load(open(args.eval_input_file, "rb"))
    metric = load_metric(path="./metric/rouge.py")
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    main()

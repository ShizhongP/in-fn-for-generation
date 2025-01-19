from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
import argparse
import pickle
from rouge_score import rouge_scorer

# from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
import numpy as np


def eval(model, tokenizer, test_dataset):
    # 创建 ROUGE 计算器
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    # 初始化 ROUGE 和 BLEU 值的累加器
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    # bleu_scores = []

    # 遍历数据集中的每个样本
    for sample in test_dataset:
        input_text = sample["document"]
        reference_summary = sample["summary"]
        # print(reference_summary)
        # 编码输入文本
        inputs = tokenizer("summarize: " + input_text, return_tensors="pt")

        try:
            # 生成摘要
            outputs = model.generate(**inputs, max_length=128)
            generated_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # print(generated_summary)
            # 计算 ROUGE 值
            scores = scorer.score(
                target=reference_summary, prediction=generated_summary
            )

            # 累加 ROUGE 值
            rouge1_scores.append(scores["rouge1"].fmeasure)
            rouge2_scores.append(scores["rouge2"].fmeasure)
            rougeL_scores.append(scores["rougeL"].fmeasure)

            # 计算 BLEU 值
            # reference_tokens = [ref.split() for ref in [reference_summary]]
            # generated_tokens = generated_summary.split()
            # bleu_score = sentence_bleu(reference_tokens, generated_tokens)

            # # 累加 BLEU 值
            # bleu_scores.append(bleu_score)

        except Exception as e:
            print(f"Error processing sample: {e}")
            continue

    # 计算平均 ROUGE 和 BLEU 值
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0
    # avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0

    # 打印平均 ROUGE 和 BLEU 值
    # print(f"Average ROUGE-1: {avg_rouge1}")
    # print(f"Average ROUGE-2: {avg_rouge2}")
    # print(f"Average ROUGE-L: {avg_rougeL}")

    return avg_rouge1, avg_rouge2, avg_rougeL


def main():

    data_path = args.eval_data_path
    data_set = pickle.load(open(data_path, "rb"))
    rouge_1 = []

    if args.eval_all:
        print("====================== evaluate all finetuned models")
        base_model_dir = "output/finetune/checkpoint-7500"
        model = AutoModelForSeq2SeqLM.from_pretrained("output/finetune/checkpoint-7500")
        tokenizer = AutoTokenizer.from_pretrained("output/finetune/checkpoint-7500")
        avg_rouge1, avg_rouge2, avg_rougeL = eval(model, tokenizer, data_set)
        print(f"Average ROUGE-1: {avg_rouge1}")
        print(f"Average ROUGE-2: {avg_rouge2}")
        print(f"Average ROUGE-L: {avg_rougeL}")

        print("=======================evaluate loo most==================")
        rouge_1 = []
        rouge_2 = []
        rouge_l = []
        for idx in range(0, 9):
            base_model_dir = "output/loo_most_{}/checkpoint-6750"
            model_dir = base_model_dir.format(idx)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            rouge_1_, rouge_2_, rouge_l_ = eval(
                model, tokenizer, data_set.select([idx])
            )
            rouge_1.append(rouge_1_)
            rouge_2.append(rouge_2_)
            rouge_l.append(rouge_l_)

        print(f"average rouge1:{np.mean(rouge_1)}")
        print(f"average rouge2:{np.mean(rouge_2)}")
        print(f"average rougeL:{np.mean(rouge_l)}")

        print("==========================evaluate loo least ===============")
        rouge_1 = []
        rouge_2 = []
        rouge_l = []
        for idx in range(0, 9):
            base_model_dir = "output/loo_least_{}/checkpoint-6750"
            model_dir = base_model_dir.format(idx)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            rouge_1_, rouge_2_, rouge_l_ = eval(
                model, tokenizer, data_set.select([idx])
            )
            rouge_1.append(rouge_1_)
            rouge_2.append(rouge_2_)
            rouge_l.append(rouge_l_)

        print(f"average rouge1:{np.mean(rouge_1)}")
        print(f"average rouge2:{np.mean(rouge_2)}")
        print(f"average rougeL:{np.mean(rouge_l)}")
        exit()

    idx = args.eval_data_idx
    # print(type(data_set))
    data = data_set.select([int(idx)])
    print(f"=============evaluate {idx} sample =======================")
    if args.eval:
        print(
            "=====================evaluate one finetuned model ======================="
        )
        model = AutoModelForSeq2SeqLM.from_pretrained("output/finetune/checkpoint-7500")
        tokenizer = AutoTokenizer.from_pretrained("output/finetune/checkpoint-7500")
        eval(model, tokenizer, data)

    if args.eval_loo_most:
        print("====================== evaluate loo most")
        base_model_dir = "output/loo_most_{}/checkpoint-6750"
        model_dir = base_model_dir.format(idx)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        eval(model, tokenizer, data)

    if args.eval_loo_least:
        print("====================== evaluate loos least")
        base_model_dir = "output/loo_least_{}/checkpoint-6750"
        model_dir = base_model_dir.format(idx)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        eval(model, tokenizer, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate leave one out ")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval_all", action="store_true")
    parser.add_argument("--eval_data_idx", type=str)
    parser.add_argument("--eval_data_path", type=str)

    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval_loo_most", action="store_true")
    parser.add_argument("--eval_loo_least", action="store_true")

    args = parser.parse_args()

    main()

import argparse
import json
import os
import random
import re
import time
from datetime import datetime, timedelta

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import openai
# from openai import OpenAI

from tqdm import tqdm

from tilse.data.timelines import Timeline as TilseTimeline
# from tilse.data import timelines
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor

from keywords_mapping import TARGET_KEYWORDS
from data import ConstrainedDataset, get_average_summary_length

from generate_constrained_events_parallel import get_target_corpus

random.seed(0)

_OPENAI_API_KEY=None # provide your API Key here


PROMPT = """
### Instruction
Using the articles about {keyword} above, please create a concise timeline with {num_event} events following the constraint below. Using only the information from the articles, provide the date and a {sen_len}-sentence summary for each important event.

### Constraint
{constraint}

### Format
YYYY-MM-DD: One-sentence Summary
YYYY-MM-DD: One-sentence Summary

### Answer
"""

def completion_with_llm(model, prompt_str, temperature=0., max_len=512, stop_tokens=None, answer_delimiter="### Answer"):
    if 'instruct' in model.lower() or 'chat' in model.lower() or 'gpt' in model.lower():
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_str}]
        completion = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=max_len,
            temperature=temperature,
            stop=stop_tokens,
        )
        logprob = completion['choices'][0].get('logprobs', None)
        if logprob is not None:
            logprob = logprob['token_logprobs']
        return completion['choices'][0]["message"]["content"].strip(), completion
    else:
        completion = openai.Completion.create(model=model,
                                        prompt=prompt_str,
                                        max_tokens=max_len,
                                        temperature=temperature,
                                        stop=stop_tokens,
                                        logprobs=True,
                                        )
        completion['choices'][0]['logprobs']['token_logprobs']
        return completion['choices'][0]['text'].strip(), completion


num2words = {1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', \
             6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine', 10: 'Ten', \
            11: 'Eleven', 12: 'Twelve', 13: 'Thirteen', 14: 'Fourteen', \
            15: 'Fifteen', 16: 'Sixteen', 17: 'Seventeen', 18: 'Eighteen', \
            19: 'Nineteen', 20: 'Twenty', 30: 'Thirty', 40: 'Forty', \
            50: 'Fifty', 60: 'Sixty', 70: 'Seventy', 80: 'Eighty', \
            90: 'Ninety', 0: 'Zero'}

def n2w(n):
    try:
        return num2words[n]
    except KeyError:
        return num2words[n-n%10] + num2words[n%10].lower()


def process_tl(args, articles, col, model, keyword, const_data):
    tokenizer1 = AutoTokenizer.from_pretrained('Xenova/gpt-4')
    tokenizer2 = AutoTokenizer.from_pretrained('/home/llm2/models/llama3.1/Meta-Llama-3.1-8B')

    cons_idx, constraint = const_data
    timelines = col.all_timelines[str(cons_idx)]
    if len(timelines) == 0:
        return
    for tl_index, gt_timeline in enumerate(timelines):
        num_event = n2w(len(gt_timeline.times))
        summary_length = n2w(get_average_summary_length(TilseTimeline(gt_timeline.time_to_summaries)))
        prompt = PROMPT.format(keyword=keyword, constraint=constraint, num_event=num_event.lower(), sen_len=summary_length.lower())
        print('\n========\nPROMPT:', prompt)
        num_tokens1 = len(tokenizer1(prompt)['input_ids'])
        num_tokens2 = len(tokenizer2(prompt)['input_ids'])
        random.seed(args.seed)
        random.shuffle(articles)
        filtered_articles = []
        indices = []
        for art_idx, article in articles:
            article_str = article["content"].strip() + "\n#################\n"
            article_len1 = len(tokenizer1(article_str)['input_ids'])
            article_len2 = len(tokenizer2(article_str)['input_ids'])
            if (num_tokens1 + article_len1 < args.context_length) and \
                (num_tokens2 + article_len2 < args.context_length):
                indices.append(article["index"])
                filtered_articles.append(article)
                num_tokens1 += article_len1
                num_tokens2 += article_len2
        article_prompt = ""
        for article in sorted(filtered_articles, key=lambda x: x["date"]):
            article_prompt += article["content"].strip() + "\n#################\n"
        prompt = article_prompt + prompt
        output, completion = completion_with_llm(prompt_str=prompt, temperature=args.temp, max_len=2048, stop_tokens=None, model=model)

        save_dir = os.path.join(args.extraction_path, args.model.split('/')[-1], f'{keyword.replace(" ", "_")}', cons_idx)
        os.makedirs(save_dir, exist_ok=True)
        print(output)
        timestamp = datetime.now().replace(microsecond=0).isoformat()
        save_path = os.path.join(save_dir, timestamp + '.txt')
        with open(save_path, "w") as out:
            out.write(output)
            
    with open(os.path.join(save_dir, 'indices.json'), 'w') as out:
        json.dump(indices, out)
                # json.dump(completion, out)


def main(args):
    print(args)
    model = args.model
    start_time = time.time()
    dataset_path = os.path.join(args.ds_path, args.dataset)
    dataset = ConstrainedDataset(dataset_path)
    collections = dataset.collections
    corpus_path = os.path.join(args.corpus_path, args.dataset, 'articles.jsonl')
    corpus = []
    key_set = set()
    with open(corpus_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            key = item['title'] + item['date'] + item['keyword']
            if key in key_set:
                continue
            corpus.append(item)
            key_set.add(key)
    print(f"Total corpus length: {len(corpus)}")
    with open(args.constraint_data) as f:
        all_constraints = json.load(f)

    if 'gpt' not in args.model.lower():
        openai.api_key = "none"  # vLLM server is not authenticated
        openai.api_base = f"http://{args.host}:{args.port}/v1"
        API_BASE = openai.api_base
    else:
        openai.api_key = _OPENAI_API_KEY
    if len(args.keywords) > 0:
        keywords = [(k, i) for i, k in enumerate(args.keywords)]
    else:
        keywords = TARGET_KEYWORDS[args.dataset]
    for keyword, index in keywords:
        if args.start_idx is not None and index < args.start_idx:
            continue
        if args.end_idx is not None and index > args.end_idx:
            break
        
        articles = list(enumerate(get_target_corpus(corpus, keyword)))
        print(f"Total articles length for {keyword}: {len(articles)}")

        under_key = keyword.replace(" ", "_")
        constraints = all_constraints[under_key]
        col = collections[under_key]
            
        with ThreadPoolExecutor(max_workers=5) as executor:
            tqdm(executor.map(lambda item: process_tl(args, articles, col, model, keyword, item), constraints.items()), total=len(articles))    

    duration = time.time() - start_time
    print(f"Duration: {duration}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--extraction_path", type=str, default="./direct_baseline/event_output")
    parser.add_argument("--corpus_path", type=str, default="./corpus")
    parser.add_argument("--dataset", type=str, default='crest')
    parser.add_argument("--ds_path", type=str, default="../data/")
    parser.add_argument("--host", type=str, default='0.0.0.0')
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--temp", type=float, default=0.)
    parser.add_argument("--start_idx", type=int, default=None)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--context_length", type=int, default=125000)
    parser.add_argument("--constraint_data", type=str, required=True)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--keywords", nargs="+", default=[])
    parser.add_argument("--parallel_n", type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
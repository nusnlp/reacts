import argparse
import json
import os
import random
import re
import time
from datetime import datetime, timedelta

import numpy as np
import openai
import requests
# from openai import OpenAI

from tqdm import tqdm

from tilse.data import timelines
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor

from keywords_mapping import TARGET_KEYWORDS
from data import Dataset

_OPENAI_API_KEY=None # provide your API Key here

SELF_REFLECT_PROMPT = """
Review the timestamped event description related to {keyword}, accompanied by a constraint. Please determine whether the event description complies with or corresponds to the constraint. Respond with 'Yes' if the event description aligns with the constraint, or with 'No' if it does not.
#################
### Event
2003-11-19: Receives the National Book Foundation Medal for Distinguished Contribution to American Letters.

### Constraint
Focus on Stephen King's awards, honors, or recognitions.

### Answer
Yes

#################
### Event
2003-11-19: Receives the National Book Foundation Medal for Distinguished Contribution to American Letters.

### Constraint
Focus on Stephen King's book releases.

### Answer
No

#################
### Event
{event}

### Constraint
{constraint}
### Answer
"""

CONSTRAINT_PROMPT_ENTITIES = """
### Instruction
Review the news article associated with the provided keyword and constraint. If the article's content does not relate to the keyword and specified constraint, output 'None'. Otherwise, summarize the most significant event related to the keyword while adhering to the constraint.

### Format
YYYY-MM-DD: One-sentence Summary

#################
### Keyword
Stephen King

### Constraint
Focus on Stephen King's book releases. 

### Content
Title: J.J. Abrams admits he was 'relieved' Stephen King wrote every episode of 'Lisey's Story'
Publish Date: 2021-06-04
Content:

J.J. Abrams said it's been "a joy" working with his childhood hero, legendary horror novelist Stephen King, to bring "Lisey's Story" to Apple TV+, out now.
Adapted from a 2006 King novel of the same name, the series tells the story of Lisey Landon (played by Julianne Moore), a widow sorting out the estate of her novelist late husband Scott (played by Clive Owen) while dealing with complicated memories of their marriage.
Abrams told Insider during a press junket for "Lisey's Story" back in May that King is "obviously such a brilliant and insanely prolific artist, but he's also just one of the loveliest, funniest, and sweetest people."
The director added that he was "flattered and thrilled," but also a little "relieved" that King decided to write every episode in the eight-part limited series thriller.
"I felt like that was going to be the thing that would help give us the roadmap we would need to do it justice," Abrams said.
The "Lost" creator took on more of a "problem-solving" role as the executive producer in the project, "dealing with things when they would come up," he added.
Abrams' fascination with King, however, began long before the two worked together on "Lisey's Story."
In 2006, Abrams and his "Lost" co-executive producers, Carlton Cuse and Damon Lindelof, met King in Maine where the four talked about the popular ABC drama and the process that goes into creating books and TV shows.
While Cuse and Lindelof each brought something for King to autograph, Abrams told Entertainment Weekly he didn't bring anything because he didn't realize it was an option.
"If I could go back in time, I probably would have found, if I still have it, the paperback to 'The Dead Zone' that I read when I was in junior high school," Abrams told Insider about rectifying his missed opportunity. "Just would be cool to have him sign that."

King says 'Lisey's Story' was inspired by his personal life
The book on which "Lisey's Story" is based is "very close" to King's heart, the writer said in a May 26 video shared on Apple TV Plus' YouTube account, "I had pneumonia around the year 2000 and came really close to stepping out," King added, noting his near-fatal brush with the lung infection.
"When I came home from the hospital, my wife had cleaned out my study and I thought to myself, 'I've died. I'm a ghost,'" he continued. "And the idea for 'Lisey's Story' came from that. Particularly the idea that writers, when they make things up. They go to a different world."

### Related Event Summary
None.

#################
### Keyword
Stephen King

### Constraint
Focus on Stephen King's involvement in television and streaming projects.

### Content
Title: J.J. Abrams admits he was 'relieved' Stephen King wrote every episode of 'Lisey's Story'
Publish Date: 2021-06-04
Content:

J.J. Abrams said it's been "a joy" working with his childhood hero, legendary horror novelist Stephen King, to bring "Lisey's Story" to Apple TV+, out now.
Adapted from a 2006 King novel of the same name, the series tells the story of Lisey Landon (played by Julianne Moore), a widow sorting out the estate of her novelist late husband Scott (played by Clive Owen) while dealing with complicated memories of their marriage.
Abrams told Insider during a press junket for "Lisey's Story" back in May that King is "obviously such a brilliant and insanely prolific artist, but he's also just one of the loveliest, funniest, and sweetest people."
The director added that he was "flattered and thrilled," but also a little "relieved" that King decided to write every episode in the eight-part limited series thriller.
"I felt like that was going to be the thing that would help give us the roadmap we would need to do it justice," Abrams said.
The "Lost" creator took on more of a "problem-solving" role as the executive producer in the project, "dealing with things when they would come up," he added.
Abrams' fascination with King, however, began long before the two worked together on "Lisey's Story."
In 2006, Abrams and his "Lost" co-executive producers, Carlton Cuse and Damon Lindelof, met King in Maine where the four talked about the popular ABC drama and the process that goes into creating books and TV shows.
While Cuse and Lindelof each brought something for King to autograph, Abrams told Entertainment Weekly he didn't bring anything because he didn't realize it was an option.
"If I could go back in time, I probably would have found, if I still have it, the paperback to 'The Dead Zone' that I read when I was in junior high school," Abrams told Insider about rectifying his missed opportunity. "Just would be cool to have him sign that."

King says 'Lisey's Story' was inspired by his personal life
The book on which "Lisey's Story" is based is "very close" to King's heart, the writer said in a May 26 video shared on Apple TV Plus' YouTube account, "I had pneumonia around the year 2000 and came really close to stepping out," King added, noting his near-fatal brush with the lung infection.
"When I came home from the hospital, my wife had cleaned out my study and I thought to myself, 'I've died. I'm a ghost,'" he continued. "And the idea for 'Lisey's Story' came from that. Particularly the idea that writers, when they make things up. They go to a different world."

### Related Event Summary
2021-06-04: The miniseries “Lisey’s Story,” adapted by King and based on his 2006 novel of the same name, premieres on Apple TV+. 

#################
### Keyword
{keyword}

### Constraint
{constraint}

### Content
{content}

### Related Event Summary
"""


def parse_timeline_obj(timeline_obj):
    constraint = timeline_obj['constraint']
    gt_events = timeline_obj['events']
    dates_to_summaries = {}
    for e in gt_events:
        if e['timestamp_level'] != 'day':
            continue
        date = datetime.strptime(e['timestamp'], '%Y-%m-%d').date()
        content = e['content']
        if date not in dates_to_summaries:
            dates_to_summaries[date] = []
        dates_to_summaries[date].append(content)
    gt_timeline = timelines.Timeline(dates_to_summaries)
    return gt_timeline, constraint


def completion_with_llm(model, prompt_str, temperature=0., max_len=512, stop_tokens=['4. '], answer_delimiter="### Answer", host=None, trial=3):
    if 'instruct' in model.lower() or 'chat' in model.lower() or 'gpt' in model.lower():
        messages_parts = prompt_str.split('#################')
        messages = [{"role": "user", "content": messages_parts[0]}]
        for msg_part in messages_parts[1:]:
            _parts = msg_part.split(answer_delimiter)
            messages.append({"role": "user", "content": _parts[0] + "### Answer"})
            messages.append({"role": "assistant", "content": _parts[1]})
        num_trial = 0
        complete = False
        while not complete:
            try:
                completion = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_len,
                    temperature=temperature,
                    stop=stop_tokens,
                )
                complete = True
            except Exception as e:
                print(e)
                num_trial += 1
                time.sleep(2.5)
                if num_trial >= trial:
                    completion = {'choices':[{"message":{"content": "None."}, 'logprobs': None}]}
                    complete = True
                    break
        
        logprob = completion['choices'][0].get('logprobs', None)
        if logprob is not None:
            logprob = logprob['token_logprobs']
        return completion['choices'][0]["message"]["content"].strip(), logprob
    else:
        
        if host is not None:
            data = {
                "model":model,
                "prompt": str(prompt_str),
                "max_tokens": str(max_len),
                "temperature": temperature,
                "stop": stop_tokens,
                "logprobs": True,
                }
            r = requests.post(host, json=data, headers={"Content-Type": "application/json"})
            print(r.text)
            completion = r.json()
        else:
            completion = openai.Completion.create(model=model,
                                        prompt=prompt_str,
                                        max_tokens=max_len,
                                        temperature=temperature,
                                        stop=stop_tokens,
                                        logprobs=False,
                                        )
        logprob = completion['choices'][0].get('logprobs', None)
        if logprob is not None:
            logprob = logprob['token_logprobs']
        return completion['choices'][0]['text'].strip(), logprob


def get_target_corpus(corpus, keyword):
    target_corpus = [item for item in corpus if item['keyword'] == keyword]
    return target_corpus


def process_article(item, constraint_str, template, args):
    title = item.get('webTitle', item.get('title', ""))
    content = item['content']
    date = item['date']
    article_index = item['index']
    tokens = tokenizer(content)['input_ids']

    if len(tokens) > args.context_length:
        content = tokenizer.decode(tokens[:args.context_length], skip_special_tokens=True)
    
    prompt_str = template.format(title=title, publish_date=date, content=content, keyword=keyword, constraint=constraint_str)
    results = []
    none_count = 0.0
    host = f"http://{args.host}:{args.port}/v1/completions" if args.use_requests else None
    for r_i in range(args.num_sampling):
        completion, logprobs = completion_with_llm(args.model, prompt_str=prompt_str, temperature=args.temp, max_len=256, stop_tokens=['#############', '\n4.', '4. ', '\n\n'], answer_delimiter="### Related Event Summary", host=host)
        sen_prob = 0 if (logprobs is None or len(logprobs) == 0) else sum(logprobs)/len(logprobs)
        output = completion
        if output.lower().strip().startswith('none'):
            none_count += 1
            if not args.self_reflect:
                results.append((sen_prob, output))
            if random.random() < 0.1:
                print(output)
        else:
            if args.self_reflect:
                reflect_prompt_str = SELF_REFLECT_PROMPT.format(event=output, keyword=keyword, constraint=constraint_str)    
                completion, _ = completion_with_llm(args.model, prompt_str=reflect_prompt_str, temperature=args.temp, max_len=256, stop_tokens=['#############', '\n4.', '4. ', '\n\n'], host=host)
                self_ref_out = completion
                if random.random() < 0.1:
                    print(constraint_str, '\n', output)
                    if self_ref_out is not None:
                        print(f'self-reflect: {output} -> {self_ref_out}')
                if completion.lower().startswith('no'):
                    output = 'None.'
                    none_count += 1
                else:
                    results.append((sen_prob, output))
            else:
                results.append((sen_prob, output))
                self_ref_out = None
                if random.random() < 0.1:
                    print(output)
    
    if len(results) > 0:
        output = sorted(results, reverse=True)[0][1]
    else:
        output = 'None.'
    
    json_obj = {
        'llm': output,
        'keyword': keyword,
        'title': title,
        'constraint': constraint_str,
        'date': date,
        'index': article_index,
    }
    return json_obj


def main(args):
    print(args)
    model = args.model
    start_time = time.time()
    if 'gpt-4' in args.model:
        tokenizer = AutoTokenizer.from_pretrained('Xenova/gpt-4')
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)

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
    for keyword, index in TARGET_KEYWORDS[args.dataset]:
        if args.start_idx is not None and index < args.start_idx:
            continue
        if args.end_idx is not None and index > args.end_idx:
            break
        
        articles = get_target_corpus(corpus, keyword)
        print(f"Total articles length for {keyword}: {len(articles)}")

        template = CONSTRAINT_PROMPT_ENTITIES
        constraints = all_constraints[keyword.replace(" ", "_")]
        for cons_idx, constraint in constraints.items():

            save_dir = os.path.join(args.extraction_path, args.model.split('/')[-1], f'{keyword.replace(" ", "_")}', cons_idx)

            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, f"{keyword.replace(' ','_')}_events.jsonl")

            with ThreadPoolExecutor(max_workers=args.parallel_n) as executor:
                save_objects = list(tqdm(executor.map(lambda item: process_article(item, constraint, template, args), articles), total=len(articles)))    
            save_objects.sort(key=lambda x: x['index'])

            with open(save_path, "w") as f:
                # json.dump(fp=f, obj=save_objects)
                for item in save_objects:
                    f.write(json.dumps(item) + '\n')
    duration = time.time() - start_time
    print(f"Duration: {duration}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--extraction_path", type=str, default="./baseline_exp/event_output")
    parser.add_argument("--corpus_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default='crest')
    parser.add_argument("--host", type=str, default='0.0.0.0')
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--temp", type=float, default=0.)
    parser.add_argument("--start_idx", type=int, default=None)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--context_length", type=int, default=1300)
    parser.add_argument("--constraint_data", type=str, required=True)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--keyword", type=str, default=None)
    parser.add_argument("--parallel_n", type=int, default=1)
    parser.add_argument("--num_sampling", type=int, default=1)
    parser.add_argument('--self_reflect', default=False, action='store_true')
    parser.add_argument('--use_requests', default=False, action='store_true')
    
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_arguments()
    main(args)
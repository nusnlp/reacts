import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import operator
import os
import pickle
import random
import re
import time
from datetime import datetime, timedelta
from pprint import pprint

import numpy as np
import openai
from langchain.embeddings import (HuggingFaceEmbeddings,
                                  SentenceTransformerEmbeddings)
from langchain.schema import Document
from langchain.vectorstores import FAISS, Chroma
from sklearn import metrics
from sklearn.metrics import normalized_mutual_info_score
from tilse.data.timelines import GroundTruth as TilseGroundTruth
from tilse.data.timelines import Timeline as TilseTimeline
from tilse.evaluation import rouge
from tqdm import tqdm
from tilse.data import timelines

import outlines

from keywords_mapping import TARGET_KEYWORDS

_OPENAI_API_KEY=None # provide your API Key here


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


def extract_all_event_details(text):
    pattern = r"(\d{4}-\d{2}-\d{2}): ([^\n]+)"
    matches = re.findall(pattern, text)
    if len(matches) == 0:
        return []
    extracted_events = []
    for d, e in matches:
        try:
            # just to check whether the date is correctly formatted. If not, attempt a correction below
            datetime.strptime(d, '%Y-%m-%d').date()
        except ValueError:
            try:
                comp = d.split('-')
                new_date = comp[0:1]
                # try assigning missing date/month
                for date_part in comp[1:]:
                    if int(date_part) == 0:
                        new_date.append('01')
                    else:
                        new_date.append(date_part)
                _ = datetime.strptime('-'.join(new_date), '%Y-%m-%d').date()
                new_date = '-'.join(new_date)
                print(f'date successfully modified from {d} to {new_date}')
                d = new_date
            except ValueError:
                print(f'[WARNING] failed to fix ill-formatted date {d}')
                continue
        item = {
            'timestamp': d,
            'text': e,
            'timestamped_text': f"{d}: {e}"
        }
        extracted_events.append(item)
    return extracted_events

def sort_events_by_date(events):
    try:
        return sorted(events, key=lambda x: x[0])
    except:
        return sorted(events, key=lambda x: x['timestamp'])

def init_vector_db(docs, embedding_func):
    """
    Initializes the vector database with document embeddings.
    Returns the initialized database.
    """
    db = Chroma.from_documents(docs, embedding_func, collection_metadata={"hnsw:space": "cosine"})
    return db



def completion_with_chat_llm(messages, temperature=0., max_len=512, stop_tokens=['4. '], model="/home/llm2/models/llama3/llama-3-8b-instruct"):
    completion = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=max_len,
        temperature=temperature,
        stop=stop_tokens,
    )
    return completion

def completion_with_llm(prompt_str, temperature=0., max_len=512, stop_tokens=[], model="meta-llama/Llama-2-13b"):
    completion = openai.Completion.create(model=model,
                                        prompt=prompt_str,
                                        max_tokens=max_len,
                                        temperature=temperature,
                                        stop=stop_tokens,
                                      )
    
    return completion


def if_two_events_same(event1, event2, keyword, model, demonstrations, temperature=0., context=None):
    """
    Determines if two events are the same.
    """
    
    if type(demonstrations) == list:
        # assert is_chat_model, "Only chat model use messages for demo."
        messages = demonstrations.copy()
        prompt_str = f"# Keyword\n{keyword}\n# Event 1\n{event1}\n# Event 2\n{event2}\n# Answer\n"
        messages.append({"role": "user", "content": prompt_str})
        completion = completion_with_chat_llm(messages, max_len=4, stop_tokens=['#############', '<|eot_id|>'], model=model)
        response = completion['choices'][0]['message']['content'].strip()
    elif type(demonstrations) == str:
        # assert is_chat_model == False, "Non-chat model use string for demo."
        prompt_str = f"{demonstrations}\n# Keyword\n{keyword}\n# Event 1\n{event1}\n# Event 2\n{event2}\n# Answer\n"
        completion, response = None, None
        completion = completion_with_llm(prompt_str, temperature, 2, stop_tokens=['\n----'], model=model)
        response = completion['choices'][0]['text']
    if random.random() < 0.5:
        print(response)
    if 'yes' in response.lower():
        return True, prompt_str
    return False, prompt_str



def event_to_doc(events, keyword, constraint=""):
    docs = []
    for idx, e in enumerate(events):
        doc = Document(
                page_content=e['timestamped_text'],
                metadata={
                    'timestamp': e['timestamp'],
                    'content': e['text'],
                    'keyword': keyword,
                    'constraint': constraint,
                    'id': idx,
                }
            )
        docs.append(doc)
    return docs


def preprocess_timeline_docs(docs, timeline):
    times = list(timeline.dates_to_summaries.keys())
    start, end = times[0], times[-1]
    timeline_docs = []
    for doc in docs:
        doc_date = doc.metadata['timestamp']
        try:
            if datetime.strptime(doc_date, '%Y-%m-%d').date() < start:
                continue
            if datetime.strptime(doc_date, '%Y-%m-%d').date() > end:
                continue
            timeline_docs.append(doc)
        except Exception as e:
            print(f"{doc_date} cannot be formatted correctly.", e)
            continue
    return timeline_docs


def generate_event_pool(docs, db, top_n, time_windows, keyword, demonstrations, constraint, output_path, model, embedding_func=None, incremental=False):
    event_pool = {}
    event2cluster = {}
    compare_res = []
    start_index = 0
    assert demonstrations is not None

    print("top_n: ", top_n)
    if len(docs) == 0:
        
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "event_pool.pickle"), 'wb') as handle:
            pickle.dump(event_pool, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(os.path.join(output_path, "event2cluster.pickle"), 'wb') as handle:
            pickle.dump(event2cluster, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(os.path.join(output_path, "compare_res.json"), 'w') as handle:
            json.dump(fp=handle, obj=compare_res, indent=4)
        
        return event_pool, event2cluster
    
    if not incremental:   # If not incremental clustering
        top_n = top_n + 1 # Retrieve top_n neighbors (+1 because the most similar one is the same as query string)
    else:
        assert embedding_func is not None
        try:
            db.delete_collection()
        except:
            pass
        db = Chroma.from_documents(docs[:1], embedding_func, collection_metadata={"hnsw:space": "cosine"})
        first_doc_id = docs[0].metadata['id']
        event_pool[-1] = [first_doc_id]
        event2cluster[first_doc_id] = -1
        start_index = 1

    for item in tqdm(docs[start_index:], desc=keyword):
        query = item.page_content
        timestamp = item.metadata['timestamp']

        TOPK = min(len(db.get()['ids']), top_n)

        event_neibors = db.similarity_search_with_relevance_scores(query=query, k=TOPK)
        event_neibors = [d for d in event_neibors if d[0].page_content != query]

        # Remove out-of-timewindow events
        has_same = False
        try:
            event_date = datetime.strptime(timestamp, '%Y-%m-%d').date()
        except ValueError:
            continue
               
        event_id = item.metadata['id']
        tmp = event_neibors.copy()
        event_neibors = []
        for doc, score in tmp:
            neighbor_date = datetime.strptime(doc.metadata['timestamp'], '%Y-%m-%d').date()
            date_diff = abs(event_date - neighbor_date) <= timedelta(days=time_windows)     # If not two events within the time window, skip.
            if not date_diff:
                continue
            event_neibors.append((doc, score))

        cls_id = -1

        for doc, score in event_neibors:
            cls_id = event2cluster.get(event_id, -1)
            event_near = doc.page_content
            neighbor_date = datetime.strptime(doc.metadata['timestamp'], '%Y-%m-%d').date()
            event_near_id = doc.metadata['id']
            neighbor_cls_id = event2cluster.get(event_near_id, -1)
            if neighbor_cls_id == cls_id and neighbor_cls_id != -1:
                continue

            event1, event2 = query, event_near
            res, prompt_str = False, None
            res, prompt_str = if_two_events_same(event1, event2, keyword, model, temperature=0., demonstrations=demonstrations)

            compare_res.append({
                'prompt': prompt_str,
                'event1': event1,
                'event2': event2,
                'constraint': constraint,
                'keyword': keyword,
                'result': res
            })
            if not res:
                continue

            if cls_id == -1 and neighbor_cls_id == -1:
                new_cls_id = max(list(event_pool.keys()) or [-1]) + 1
                event_pool[new_cls_id] = [event_id, event_near_id]
                event2cluster[event_id] = new_cls_id
                event2cluster[event_near_id] = new_cls_id
            elif cls_id == -1 and neighbor_cls_id != -1:
                event_pool[neighbor_cls_id] = event_pool[neighbor_cls_id] + [event_id]
                event2cluster[event_id] = neighbor_cls_id
            elif cls_id != -1 and neighbor_cls_id == -1:
                event_pool[cls_id] = event_pool[cls_id] + [event_near_id]
                event2cluster[event_near_id] = cls_id
            elif cls_id != -1 and neighbor_cls_id != -1 and cls_id != neighbor_cls_id:
                event_pool[cls_id] = list(set(event_pool[cls_id] + event_pool[neighbor_cls_id]))
                del event_pool[neighbor_cls_id]
                for e_id in event_pool[cls_id]:
                    event2cluster[e_id] = cls_id
            else:
                pass
            has_same = True

        if not has_same and cls_id == -1:
            new_cls_id = max(list(event_pool.keys()) or [-1]) + 1
            event_pool[new_cls_id] = [event_id]
            event2cluster[event_id] = new_cls_id

        if incremental:
            db.add_documents([item])
        
    # if incremental:
    db.delete_collection()
    
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "event_pool.pickle"), 'wb') as handle:
        pickle.dump(event_pool, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(os.path.join(output_path, "event2cluster.pickle"), 'wb') as handle:
        pickle.dump(event2cluster, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(os.path.join(output_path, "compare_res.json"), 'w') as handle:
        json.dump(fp=handle, obj=compare_res, indent=4)
    
    return event_pool, event2cluster


def main(args):
    random.seed(args.seed)
    if 'gpt' in args.model.lower():
        openai.api_key = _OPENAI_API_KEY
    else:
        openai.api_key = "none"
        openai.api_base = f"http://{args.host}:{args.port}/v1"
        API_BASE = openai.api_base
    with open(args.fewshot, 'r') as f:
        files = json.load(fp=f)
    start_time = time.time()
    for keyword, index in TARGET_KEYWORDS[args.dataset]:
        if args.start_idx is not None and index < args.start_idx:
            continue
        if args.end_idx is not None and index > args.end_idx:
            break
        input_dir = os.path.join(args.input, keyword.replace(' ', '_'))
        for constraint_num in os.listdir(input_dir):
            keyword_results = []
            if 'all' not in args.keyword:
                if keyword not in args.keyword:
                    continue
            start_time = time.time()

            demons = files[keyword]
            if 'instruct' in args.model.lower() or 'chat' in args.model.lower() or 'gpt' in args.model.lower():
                demons_list = demons.split('\n----\n')
                demons = [{"role": "user", "content": demons_list[0]}]
                for ex in demons_list[1:]:
                    ex_parts = ex.split('# Answer', 1)
                    assert len(ex_parts) == 2, ex_parts
                    question, answer = tuple(ex_parts)
                    demons.append({"role": "user", "content": question.strip() + '# Answer\n'})
                    demons.append({"role": "assistant", "content": answer.strip()})

            event_path = os.path.join(input_dir, constraint_num, f"{keyword.replace(' ', '_')}_events.jsonl")
        
            with open(event_path, 'r') as f:
                events_file = [json.loads(x) for x in f]
            print(f"loaded {len(events_file)} input.")

            events, multi_events_indexs = [], []
            constraint = None
            for idx, item in enumerate(events_file):
                text = item['llm']
                extracted_events = extract_all_event_details(text)
                if len(extracted_events) > 1:
                    multi_events_indexs.append(idx)
                events.extend(extracted_events)
                if constraint is None:
                    constraint = item["constraint"]
                else:
                    assert constraint == item["constraint"], "Constraints are different in the same file!"
            sorted_events = sort_events_by_date(events)
            sorted_docs = event_to_doc(sorted_events, keyword, constraint)
            print(f"Loaded {len(sorted_docs)} events with date.")
            tl_docs = sorted_docs

            print(f"Loading sentence embedding model and cls model...")
            embedding_func = SentenceTransformerEmbeddings(model_name='thenlper/gte-large')

            print(f"Finish loading sentence embedding model and cls model.")


            db = None
            if not args.incremental:
                try:
                    db.dete_collection()
                except:
                    pass
                db = init_vector_db(tl_docs, embedding_func)

            tl_docs_dicts = []
            output_path = os.path.join(args.output, args.model.split('/')[-1], keyword.replace(" ", "_"))
            tl_output_path = os.path.join(output_path, constraint_num)
            os.makedirs(tl_output_path, exist_ok=True)
            for d in tl_docs:
                tl_docs_dicts.append({
                    'text': d.page_content,
                    'timestamp': d.metadata['timestamp'],
                    'id': d.metadata['id']
                })
            with open(os.path.join(tl_output_path, 'docs.pickle'), 'wb') as handle:
                pickle.dump(tl_docs_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)
            

            event_pool, event2cluster = generate_event_pool(
                                                            docs=tl_docs,
                                                            db=db,
                                                            top_n=args.top_n,
                                                            time_windows=args.time_windows,
                                                            keyword=keyword,
                                                            demonstrations = demons,
                                                            constraint=constraint,
                                                            output_path=tl_output_path,
                                                            model=args.model,
                                                            embedding_func=embedding_func,
                                                            incremental=args.incremental
                                                        )
            
    duration = time.time() - start_time
    print(f"Total running time: {duration}s.")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--keyword", type=str, nargs='*', required=True)
    parser.add_argument("--output", type=str, default="./timeline_output")
    parser.add_argument("--dataset", type=str, default="crest")
    parser.add_argument("--host", type=str, default='0.0.0.0')
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--top_n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--start_idx", type=int, default=None)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--fewshot", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--incremental", action='store_true')
    parser.add_argument("--time_windows", type=int, default=0)
    parser.add_argument("--timelines_path", type=str, required=True)
    parser.add_argument("--timeline_index", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

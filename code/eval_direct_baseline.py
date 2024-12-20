import json
import pickle
import re
import os

from tilse.data.timelines import Timeline as TilseTimeline
from tilse.data.timelines import GroundTruth as TilseGroundTruth
from tilse.evaluation import rouge
import operator
from evaluation import get_scores, evaluate_dates, get_average_results, zero_scores, full_score
from data import ConstrainedDataset, get_average_summary_length
from datetime import datetime
from pprint import pprint
from collections import Counter
from tqdm import tqdm

from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings import SentenceTransformerEmbeddings

from langchain.schema import Document
from keywords_mapping import TARGET_KEYWORDS

from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import openai
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"


embedding_func = SentenceTransformerEmbeddings(model_name='thenlper/gte-large')

def load_data(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def save_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def get_average_summary_length(ref_tl):
    lens = [len(summary) for date, summary in ref_tl.dates_to_summaries.items()]
    return round(sum(lens) / len(lens))

def text_rank(sentences, embedding_func, personalization=None):
    sentence_embeddings = embedding_func.embed_documents(texts=sentences)
    cosine_sim_matrix = cosine_similarity(sentence_embeddings)
    nx_graph = nx.from_numpy_array(cosine_sim_matrix)
    scores = nx.pagerank(nx_graph, personalization=personalization)
    return sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

def get_pairs(event_pool):
    pairs, singletons = [], []
    for cluster_id, nodes in event_pool.items():
        if len(nodes) > 1:
            pairs.extend((min(nodes[i], nodes[j]), max(nodes[i], nodes[j])) for i in range(len(nodes)) for j in range(i + 1, len(nodes)))
        else:
            singletons.append(nodes[0])
    return pairs, singletons

def get_avg_score(scores):
    return sum(scores) / len(scores)

evaluator = rouge.TimelineRougeEvaluator(measures=["rouge_1", "rouge_2"])
metric = 'align_date_content_costs_many_to_one'


def evaluate_timeline(overall_results, timeline_paths, collections, embedding_func, args):
    overall_r1_list, overall_r2_list, overall_d1_list = [], [], []

    for timeline_path in timeline_paths:
        print(f"Trial {timeline_path}:")
        trial_res = process_trial(collections, timeline_path, args, embedding_func, overall_results)
        pprint(trial_res)
        
        rouge_1 = trial_res[0]['f_score']
        rouge_2 = trial_res[1]['f_score']
        date_f1 = trial_res[2]['f_score']
        trial_save_path = os.path.join(args.output, timeline_path)
        os.makedirs(trial_save_path, exist_ok=True)

        overall_r1_list.append(rouge_1)
        overall_r2_list.append(rouge_2)
        overall_d1_list.append(date_f1)

        save_json(trial_res, os.path.join(trial_save_path, 'avg_score.json'))

    save_json(overall_results, os.path.join(args.output, 'global_result.json'))

    avg_r1 = get_avg_score(overall_r1_list)
    avg_r2 = get_avg_score(overall_r2_list)
    avg_d1 = get_avg_score(overall_d1_list)
    final_results = {
        'rouge1': avg_r1,
        'rouge2': avg_r2,
        'dateF1': avg_d1
    }
    print(final_results)

    save_json(final_results, os.path.join(args.output, 'average_result.json'))


def process_trial(collections, timeline_path, args, embedding_func, overall_results):
    results = []
    for keyword, index in tqdm(TARGET_KEYWORDS[args.dataset]):
        if args.start_idx is not None and index < args.start_idx:
            continue
        if args.end_idx is not None and index > args.end_idx:
            break
        if args.keywords is not None and keyword not in args.keywords:
            continue
        
        print(f"processing {keyword}...")
        under_key = keyword.replace(" ", "_")
        col = collections[under_key]
        for const, timelines in col.all_timelines.items():
            for tl_index, gt_timeline in enumerate(timelines):
                summary_length = get_average_summary_length(TilseTimeline(gt_timeline.time_to_summaries))
                timeline_res = process_timeline(gt_timeline, summary_length, under_key, tl_index, timeline_path, const, args, embedding_func)
                (rouge_scores, date_scores, pred_timeline_dict) = timeline_res
                results.append(timeline_res)
                print(rouge_scores)
                # Update overall results
                if keyword not in overall_results:
                    overall_results[keyword] = {}
                if const not in overall_results[keyword]:
                    overall_results[keyword][const] = {}
                if tl_index not in overall_results[keyword][const]:
                    overall_results[keyword][const][tl_index] = []
                overall_results[keyword][const][tl_index].append((rouge_scores, date_scores))
    trial_res = get_average_results(results)
    return trial_res


def load_events(path):
    with open(path, 'r') as f:
        events = [json.loads(x) for x in f]
    content2type = {}
    for e in events:
        for key in ['llm', 'sentence']:
            if key in e:
                content = e[key].split(':')[-1].strip()
                content2type[content] = key
                break
    return events, content2type


def process_timeline(gt_timeline, summary_length, keyword, tl_index, timeline_path, const, args, embedding_func):
    actual_timeline_dir = os.path.join(timeline_path, keyword, const)
    try:
        filename = sorted([f for f in os.listdir(actual_timeline_dir) if str(f).endswith('.txt')])
        tl_filepath = os.path.join(actual_timeline_dir, filename[-1]) # take last file

        with open(tl_filepath) as f:
            data = f.read()
        data = re.sub(r'\n+', '\n', data).strip()
        pred_timeline_dict = {}
        for line in data.split('\n'):
            parts = line.split(':')
            if len(parts) != 2:
                continue
            date_str, event = tuple(parts)
            try:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
                pred_timeline_dict[date_obj] = [l + '.' for l in event.split('. ')]
            except:
                pass
        pred_timeline = TilseTimeline(pred_timeline_dict)
    except:
        pred_timeline = TilseTimeline({})

    print(f'===== {keyword}: {const} =====')
    print('pred_timeline:\n', pred_timeline)
    print('ground_truth:\n', gt_timeline)
    if len(pred_timeline) == 0 and len(gt_timeline) == 0:
        rouge_scores = {}
        rouge_scores['rouge_1'] = full_score()
        rouge_scores['rouge_2'] = full_score()
        date_scores = full_score()
    elif len(pred_timeline) == 0 or len(gt_timeline) == 0:
        rouge_scores = {}
        rouge_scores['rouge_1'] = zero_scores()
        rouge_scores['rouge_2'] = zero_scores()
        date_scores = zero_scores()
    else:
        # Evaluate summarization
        ground_truth = TilseGroundTruth([TilseTimeline(gt_timeline.date_to_summaries)])
        evaluator = rouge.TimelineRougeEvaluator(measures=["rouge_1", "rouge_2"])
        rouge_scores = get_scores(metric, pred_timeline, ground_truth, evaluator)
        date_scores = evaluate_dates(pred_timeline, ground_truth)
    timeline_res = (rouge_scores, date_scores, pred_timeline)

    return timeline_res




def cluster_events(docs, pairs, top_l):
    # Build pools
    event_pool = dict()
    event2cluster = dict()
    for edge in pairs:
        event_id, event_near_id = edge[0], edge[1]
        cls_id = event2cluster.get(event_id, -1)
        neighbor_cls_id = event2cluster.get(event_near_id, -1)

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
    
    total_ids = [doc['id'] for doc in docs]
    for node in total_ids:
        if node in event2cluster:
            continue 
        new_cls_id = max(list(event_pool.keys()) or [-1]) + 1
        event_pool[new_cls_id] = [node]
        event2cluster[node] = new_cls_id
    
    
    clusters = list(event_pool.values())
    clusters.sort(key=len, reverse=True)

    docs_map = {}
    for doc in docs:
        id = doc['id']
        docs_map[id] = doc
    
    top_clusters = {}
    for idx, cluster in enumerate(clusters):
        # Get dates of events in this cluster
        dates = [docs_map[i]['timestamp'] for i in cluster]
        # Get the most often exist date
        cluster_date = max(set(dates), key=dates.count)

        if cluster_date not in top_clusters:
            top_clusters[cluster_date] = [list(cluster)]
        else:
            top_clusters[cluster_date].append(list(cluster))
        tmp = sorted(top_clusters[cluster_date], key=lambda x: len(x), reverse=True)
        top_clusters[cluster_date] = tmp
    
    for cluster_date in top_clusters.keys():
        tmp = []
        N = 1
        for i in range(N):
            tmp.extend(top_clusters[cluster_date][i])
        cnt = len(tmp)
        top_clusters[cluster_date] = (cnt, top_clusters[cluster_date][:N])

    top_clusters = dict(sorted(top_clusters.items(), key=lambda item: item[1][0], reverse=True))
    top_clusters = list(top_clusters.values())


    # Assign core event
    cluster_info = []
    exist_dates = set()
    for cnt, same_day_clusters in top_clusters:
        all_events = []
        core_events = []
        cluster_date = None
        node_cnt = 0
        for cluster in same_day_clusters:
            node_cnt += len(cluster)
            # Get dates of events in this cluster
            dates = [docs_map[i]['timestamp'] for i in cluster]
            # Get the most often exist date
            cluster_date = max(set(dates), key=dates.count)
            # Get events which have the same date as the cluster date
            same_date_events = [docs_map[i] for i in cluster if docs_map[i]['timestamp'] == cluster_date]
            all_events.append([docs_map[i] for i in cluster])
            # Get core event (event with cluster date and cluster topic)
            core_event = next((event for event in same_date_events), None)
            core_events.append(core_event['text'].split(':')[-1].strip())
        # Prepare cluster info
        if cluster_date in exist_dates:
            continue
        exist_dates.add(cluster_date)
        cluster_info.append((cluster_date, node_cnt, core_events, all_events))
        if len(cluster_info) == (top_l):
            break

    # Sort clusters by date
    cluster_info.sort(key=operator.itemgetter(0))
    return cluster_info


def main(args):
    os.makedirs(args.output, exist_ok=True)
    dataset_path = os.path.join(args.ds_path, args.dataset)
    dataset = ConstrainedDataset(dataset_path)
    collections = dataset.collections
    embedding_func = SentenceTransformerEmbeddings(model_name='thenlper/gte-large')

    evaluate_timeline({}, args.timelines_paths, collections, embedding_func, args)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timelines_paths", nargs="+", required=True)
    parser.add_argument("--keywords", nargs="+", default=None)
    parser.add_argument("--output", type=str, default="./result")
    parser.add_argument("--dataset", type=str, default="crest")
    parser.add_argument("--text_rank", action='store_true')
    parser.add_argument("--start_idx", type=int, default=None)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--ds_path", type=str, default="../data/")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

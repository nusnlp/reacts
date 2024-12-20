## REACTS
Reflective Algorithm for Constrained Timeline Summarization

### Preparation
1. Install the code dependencies.
```
pip install -r requirements.txt
```
2. Preprocess dataset articles to the required format using ```preprocess_articles.py```.
```
python preprocess_articles.py --ds_path "../data/" --dataset "crest" --save_path "./corpus"
```

3. Host the LLM server
Host the model through vLLM using the below command in a separate process (use a separate terminal session, or put the process below to the background):
```
python -m vllm.entrypoints.openai.api_server --model "meta-llama/Meta-Llama-3.1-8B" --port 8000
```

### Workflow
1. Summarize the articles with the following commands:
```
python generate_constrained_events_parallel.py \
    --dataset crest \
    --model "meta-llama/Meta-Llama-3.1-8B" \
    --constraint_data constraint_dict.json \
    --extraction_path "./event_outputs" \
    --parallel_n 50
```

If you want to use self-reflect, please provide the argument `--self_reflect`.


2. Perform incremental clustering process using `generate_clusters.py`.
```
python generate_clusters.py \
    --model meta-llama/Meta-Llama-3.1-8B \
    --output ./timelines_output \
    --input ./event_outputs/entities/Meta-Llama-3.1-8B \
    --top_n 20 \
    --dataset crest \
    --incremental \
    --keyword all
```

3. Perform cluster and sentence selection, timeline generation, and evaluation using `cluster_tls_eval.py`.
```
python cluster_tls_eval.py \
    --timelines_path ./timelines_output/entities/Llama-2-13b-hf \
    --events_path ./event_outputs/entities/Llama-2-13b-hf \
    --output ./result/entities \
    --dataset crest \
    --text_rank
```

# Search-R1 Retriever on A800

The service runs in Docker container `verl` and listens on
`http://127.0.0.1:8181`. It uses the existing eight-GPU FAISS index; do not
download another copy.

## Paths

```text
/ssd2/chengmingquan/Search-R1/retrieval_launch.sh
/ssd2/data/NQ_dataset/e5_Flat.index
/ssd2/data/NQ_dataset/wiki-18.jsonl
/ssd2/llm_models/e5-base-v2
```

## Start

```bash
docker exec -d verl bash -lc \
  'cd /ssd2/chengmingquan/Search-R1 && nohup bash ./retrieval_launch.sh \
   >/tmp/retriever.log 2>&1 < /dev/null'
```

Detached mode is required. A foreground process started by a short-lived
`docker exec` is terminated when the exec session exits.

## Check

```bash
docker exec verl bash -lc 'pgrep -af retrieval_server.py'
docker exec verl bash -lc 'tail -f /tmp/retriever.log'
docker exec verl python -c 'import faiss; print(faiss.get_num_gpus())'
```

The FAISS GPU count should be 8. Initial loading reads about 61 GB of index and
14 GB of corpus and can take several minutes. A running PID alone does not mean
the service is ready; port 8181 must respond to a request.

## Request

```bash
docker exec verl bash -lc \
  'curl -sS -m 30 -X POST http://127.0.0.1:8181/retrieve \
   -H "Content-Type: application/json" \
   -d "{\"queries\":[\"What is Python?\"],\"topk\":1}"'
```

Readiness is a successful HTTP response with retrieval JSON. If connection is
refused while the PID remains alive, continue waiting and inspect the log.

## Stop and restart

```bash
docker exec verl bash -lc \
  'pkill -f "search_r1/search/retrieval_server.py" || true'
```

Only kill GPU processes when explicitly authorized. Before restarting, verify
that the previous retrieval PID has exited and GPU memory is available.

# Search-R1 Retriever on A800

The service runs in Docker container `verl` and listens on
`http://127.0.0.1:8181`. It uses the existing eight-GPU FAISS index; do not
download another copy. Verified in-container on 2026-08-13: PID `2587285`,
`faiss.get_num_gpus() == 8`, and every request below returned the documented
result.

The container uses `NetworkMode=host`, so `127.0.0.1:8181` resolves to the same
listener from inside the container and from the A800 host. `docker port verl`
is empty and no published port mapping exists — that is expected under host
networking, not a misconfiguration.

## Paths

```text
/ssd2/chengmingquan/Search-R1/retrieval_launch.sh
/ssd2/data/NQ_dataset/e5_Flat.index
/ssd2/data/NQ_dataset/wiki-18.jsonl
/ssd2/llm_models/e5-base-v2
```

Running process arguments (from `pgrep -af retrieval_server.py`):
`--topk 3 --retriever_name e5 --faiss_gpu`. Steady-state GPU usage is about
4.4 GB per GPU (about 5.3 GB on GPU 0).

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
docker exec verl bash -lc 'tail -n 50 /tmp/retriever.log'
docker exec verl python -c 'import faiss; print(faiss.get_num_gpus())'
```

The FAISS GPU count should be 8. Initial loading reads about 61 GB of index and
14 GB of corpus and can take several minutes. A running PID alone does not mean
the service is ready; port 8181 must respond to a request. Each served request
appends a `POST /retrieve HTTP/1.1" 200 OK` line to `/tmp/retriever.log`, which
is the quickest way to confirm traffic is arriving.

Do not use `tail -f` over the relay channel; it keeps the channel open until
the command times out. Poll with `tail -n <N>` instead.

## Request

```bash
docker exec verl bash -lc \
  'curl -sS -m 30 -X POST http://127.0.0.1:8181/retrieve \
   -H "Content-Type: application/json" \
   -d "{\"queries\":[\"What is Python?\"],\"topk\":1,\"return_scores\":true}"'
```

Readiness is a successful HTTP 200 response with retrieval JSON. The verified
response includes a Python document and score `0.8638745546340942`.
Use `return_scores: true` with the current Search-R1 server implementation;
without it, the server returns HTTP 500 while unpacking the batch search return
value (reconfirmed on 2026-08-13). If connection is refused while the PID
remains alive, continue waiting and inspect the log.

## Verified API behaviour

Response shape: `result` is one list per query, each element
`{"document": {"id", "contents"}, "score": <float>}`.

| Request | Result |
| --- | --- |
| 1 query, `topk: 1`, `return_scores: true` | 200, one hit, 0.012 s |
| 2 queries, `topk: 3` | 200, `result` length 2, three hits each |
| `topk: 20` | 200, twenty hits |
| `topk` omitted | 200, three hits (server default `--topk 3`) |
| `return_scores` omitted | **500 Internal Server Error** |
| `queries: []` | 200, `{"result": []}` |
| `GET /retrieve` | 405 Method Not Allowed |
| `GET /` | 404 |
| `GET /docs` | 200 (FastAPI Swagger UI) |

Python clients work over both `urllib.request` and `requests` (the container
ships `requests` 2.32.5, which is what `verl.tools.search_tool` uses):

```bash
docker exec verl python -c 'import requests; \
  r = requests.post("http://127.0.0.1:8181/retrieve", \
    json={"queries": ["When was Amazon founded?"], "topk": 3, "return_scores": True}, \
    timeout=30); \
  print(r.status_code, len(r.json()["result"][0]))'
```

## Measured throughput

32 concurrent threads issuing 64 single-query `topk: 3` requests from inside the
container: all 200, 1.12 s wall clock, about 57 QPS, p50 0.479 s, p95 0.580 s,
max 0.634 s. `search_tool_config.yaml` sets `num_workers: 120` and
`rate_limit: 120`, which is above this measured concurrency — if rollout stalls
on search, lower both before assuming the retriever is down.

## Use from training

`scripts/Search-R1/train.sh` defaults `RETRIEVER_URL` to
`http://127.0.0.1:8181/retrieve`.
`search_tool_config.yaml` reads the same variable via
`${oc.env:RETRIEVER_URL,http://127.0.0.1:8181/retrieve}`. Point both at another
host by exporting `RETRIEVER_URL` before launching.

The launcher uses `SearchOPDMuon` with FSDP2 by default. Training uses five
rollout samples per prompt; validation and the independent final benchmark pass
use one sample. The complete training parquet is used without a train/validation
split. NQ and HotpotQA test parquets are combined for the configured validation
pass and also loaded as separate final-test dataloaders reported under
`test-core/searchR1_nq/...` and `test-core/searchR1_hotpotqa/...`. The final-test
evaluation does not mutate validation solved-UID history, but the checkpoint
selection itself is test-informed because validation uses those same test
files. Treat both result sets as diagnostic rather than independent held-out
generalization.

## Stop and restart

```bash
docker exec verl bash -lc \
  'pkill -f "search_r1/search/retrieval_server.py" || true'
```

Only kill GPU processes when explicitly authorized. Before restarting, verify
that the previous retrieval PID has exited and GPU memory is available.

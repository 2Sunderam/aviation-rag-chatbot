# Retrieval eval (90 questions: 80 single, 10 multi-part)

Relevant = same book + overlapping pages as a gold chunk. Multi-part rows have 2 gold chunks.


## all

| setup | mrr | hit@1 | hit@5 | recall@5 | recall@10 | precision@5 |
|---|---|---|---|---|---|---|
| dense | 0.750 | 0.633 | 0.911 | 0.894 | 0.956 | 0.304 |
| bm25 | 0.752 | 0.622 | 0.911 | 0.894 | 0.928 | 0.296 |
| hybrid | 0.829 | 0.744 | 0.944 | 0.922 | 0.961 | 0.318 |
| hybrid+planner | 0.835 | 0.744 | 0.956 | 0.944 | 0.961 | 0.327 |

## single

| setup | mrr | hit@1 | hit@5 | recall@5 | recall@10 | precision@5 |
|---|---|---|---|---|---|---|
| dense | 0.747 | 0.637 | 0.900 | 0.900 | 0.950 | 0.285 |
| bm25 | 0.763 | 0.637 | 0.912 | 0.912 | 0.938 | 0.287 |
| hybrid | 0.829 | 0.750 | 0.938 | 0.938 | 0.963 | 0.310 |
| hybrid+planner | 0.832 | 0.738 | 0.963 | 0.963 | 0.963 | 0.312 |

## multi

| setup | mrr | hit@1 | hit@5 | recall@5 | recall@10 | precision@5 |
|---|---|---|---|---|---|---|
| dense | 0.770 | 0.600 | 1.000 | 0.850 | 1.000 | 0.460 |
| bm25 | 0.667 | 0.500 | 0.900 | 0.750 | 0.850 | 0.360 |
| hybrid | 0.833 | 0.700 | 1.000 | 0.800 | 0.950 | 0.380 |
| hybrid+planner | 0.860 | 0.800 | 0.900 | 0.800 | 0.950 | 0.440 |
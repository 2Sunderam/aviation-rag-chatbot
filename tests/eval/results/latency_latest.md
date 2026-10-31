# Latency eval (100 questions, sequential, Mercury 3 answers)

| stage | p50 ms | p95 ms | max ms | mean ms |
|---|---|---|---|---|
| plan | 525 | 686 | 905 | 544 |
| embed | 48 | 100 | 111 | 58 |
| search_text | 20 | 24 | 26 | 20 |
| search_figures | 13 | 17 | 20 | 13 |
| llm_first_token | 578 | 764 | 966 | 597 |
| llm_total | 583 | 903 | 1318 | 634 |
| end_to_end | 1229 | 1693 | 2019 | 1269 |

- **p95 end-to-end: 1.69 s** (target <= 5 s): PASS
- answers within 5 s: 100% | within the 2-5 s band: 1% (faster than 2 s counts as fine)
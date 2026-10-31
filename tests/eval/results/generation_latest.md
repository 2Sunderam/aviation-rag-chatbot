# Generation eval (100 questions, judge: mercury-3)

| metric | value | meaning |
|---|---|---|
| ragas faithfulness | 0.982 | |
| ragas answer_relevancy | 0.913 | |
| ragas llm_context_precision_with_reference | 0.855 | |
| ragas context_recall | 0.960 | |
| hallucination rate | 0.099 | answers with >= 1 unsupported claim (faithfulness < 1) |
| severe hallucination rate | 0.0 | faithfulness < 0.5 |
| citation accuracy | 0.81 | cited chunk supports the sentence (226 pairs) |
| invalid citation rate | 0.0 | cited numbers that do not exist |
| answers without citation | 0.0 | |
| unanswerable correctly refused | 1.0 | |
| answerable wrongly refused | 0.067 | |
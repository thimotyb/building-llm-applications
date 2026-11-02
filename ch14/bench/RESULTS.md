# Judge benchmark

> Partial runs present: all backends are compared on the same 49 tickets (ids 0..48). Failed tickets are excluded from the accuracy columns and counted in the errors column.


### team

| backend | n | acc | ECE | Brier | conf | acc@c | cov@c |
|---|---|---|---|---|---|---|---|
| deepseek-v4-flash | 49 | 0.88 | 0.076 | 0.175 | 0.88 | 0.91 | 0.90 |
| gemma-12b | 49 | 0.77 | 0.233 | 0.435 | 0.92 | 0.77 | 1.00 |
| jev-1.14.0 | 49 | 0.90 | 0.077 | 0.147 | 0.91 | 0.97 | 0.82 |
| xgboost-tfidf | 49 | 0.98 | 0.077 | 0.029 | 0.92 | 1.00 | 0.94 |

### urgent

| backend | n | acc | ECE | Brier | conf | acc@c | cov@c |
|---|---|---|---|---|---|---|---|
| deepseek-v4-flash | 49 | 0.98 | 0.063 | 0.026 | 0.94 | 1.00 | 0.98 |
| gemma-12b | 49 | 0.85 | 0.096 | 0.214 | 0.90 | 0.85 | 1.00 |
| jev-1.14.0 | 49 | 0.98 | 0.132 | 0.051 | 0.87 | 1.00 | 0.94 |
| xgboost-tfidf | 49 | 0.94 | 0.091 | 0.086 | 0.90 | 0.98 | 0.92 |

### refund

| backend | n | acc | ECE | Brier | conf | acc@c | cov@c |
|---|---|---|---|---|---|---|---|
| deepseek-v4-flash | 49 | 1.00 | 0.031 | 0.005 | 0.97 | 1.00 | 1.00 |
| gemma-12b | 49 | 0.94 | 0.048 | 0.089 | 0.98 | 0.94 | 1.00 |
| jev-1.14.0 | 49 | 1.00 | 0.043 | 0.006 | 0.96 | 1.00 | 1.00 |
| xgboost-tfidf | 49 | 0.98 | 0.035 | 0.051 | 0.94 | 0.98 | 0.98 |

### Speed and cost (all three questions per ticket in one call)

| backend | errors | mean latency / ticket | total cost | note |
|---|---|---|---|---|
| deepseek-v4-flash | 0 | 2219 ms | $0.0078 | deepseek/deepseek-v4-flash |
| gemma-12b | 2 | 16073 ms | $0.0000 | PARTIAL: 49 of 120 tickets, 2 failed (counted as errors) |
| jev-1.14.0 | 0 | 194 ms | $0.0008 | jev-latest |
| xgboost-tfidf | 0 | 0 ms | $0.0000 | 5-fold CV; trained on ~80% of the labelled tickets, needs labels per task |

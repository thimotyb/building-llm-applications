# Judge benchmark

### team

| backend | n | acc | ECE | Brier | conf | acc@c | cov@c |
|---|---|---|---|---|---|---|---|
| deepseek-v4-flash | 120 | 0.87 | 0.063 | 0.153 | 0.87 | 0.93 | 0.88 |
| jev-1.14.0 | 120 | 0.91 | 0.032 | 0.114 | 0.93 | 0.97 | 0.88 |
| xgboost-tfidf | 120 | 0.86 | 0.076 | 0.196 | 0.89 | 0.91 | 0.90 |

### urgent

| backend | n | acc | ECE | Brier | conf | acc@c | cov@c |
|---|---|---|---|---|---|---|---|
| deepseek-v4-flash | 120 | 0.98 | 0.063 | 0.023 | 0.94 | 1.00 | 0.98 |
| jev-1.14.0 | 120 | 0.99 | 0.128 | 0.055 | 0.86 | 1.00 | 0.93 |
| xgboost-tfidf | 120 | 0.87 | 0.055 | 0.196 | 0.89 | 0.92 | 0.88 |

### refund

| backend | n | acc | ECE | Brier | conf | acc@c | cov@c |
|---|---|---|---|---|---|---|---|
| deepseek-v4-flash | 120 | 1.00 | 0.033 | 0.007 | 0.97 | 1.00 | 1.00 |
| jev-1.14.0 | 120 | 0.99 | 0.047 | 0.012 | 0.96 | 1.00 | 0.99 |
| xgboost-tfidf | 120 | 0.97 | 0.033 | 0.063 | 0.95 | 0.97 | 0.99 |

### Speed and cost (all three questions per ticket in one call)

| backend | errors | mean latency / ticket | total cost | note |
|---|---|---|---|---|
| deepseek-v4-flash | 0 | 2057 ms | $0.0202 | deepseek/deepseek-v4-flash |
| jev-1.14.0 | 0 | 187 ms | $0.0020 | jev-latest |
| xgboost-tfidf | 0 | 0 ms | $0.0000 | 5-fold CV; trained on ~80% of the labelled tickets, needs labels per task |

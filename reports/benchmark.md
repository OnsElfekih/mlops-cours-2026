# Benchmark de latence — german-credit, modèle v12

1000 appels par backend, après 10 appels de chauffe · x86_64 · Python 3.12.3

| Backend | P50 (ms) | P95 (ms) | P99 (ms) |
|---|---|---|---|
| sklearn_pipeline | 4.011 | 6.039 | 6.308 |
| onnx_python | 0.080 | 0.085 | 0.089 |
| onnx_batch_100 | 0.345 | 0.357 | 0.374 |

Accélération ONNX vs scikit-learn (P50, 1 ligne) : x49.9

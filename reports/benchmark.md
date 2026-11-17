# Benchmark de latence — german-credit, modèle v12

1000 appels par backend, après 10 appels de chauffe · x86_64 · Python 3.12.3

| Backend | P50 (ms) | P95 (ms) | P99 (ms) |
|---|---|---|---|
| sklearn_pipeline | 3.451 | 4.048 | 4.996 |
| onnx_python | 0.077 | 0.092 | 0.125 |
| onnx_batch_100 | 0.298 | 0.379 | 0.535 |

Accélération ONNX vs scikit-learn (P50, 1 ligne) : x44.7

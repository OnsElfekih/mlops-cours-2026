# Rapport — Lab 5 : serving avec KServe

Binôme : Eya & Ons · Dataset : german-credit · Modèle : `scoring-model` v12 ·
Seuil de décision servi : 0,3182

## 1. Résumé

L'image FastAPI + ONNX Runtime du Lab 4 est servie par un InferenceService KServe `scoring-model`
(namespace `scoring`) en mode Knative, derrière la passerelle Kourier (NodePort 30080, routage par
en-tête `Host`). L'autoscaler Knative (KPA) pilote le nombre de pods sur le débit (cible 200 req/s
par pod, de 0 à 5 pods, scale-to-zero). En fin de lab, la v2.0.0 (route `/version`) a été déployée
en canary 90/10 puis promue à 100 % (révision `scoring-model-predictor-00003`).

## 2. Benchmark de latence

Locust, requêtes `/predict` sur de vraies lignes de `test.csv` (runs 500 et 1000 avec `--processes 4`).

| Run | Requêtes | RPS | P50 (ms) | P95 (ms) | P99 (ms) | Erreurs (%) |
|---|---|---|---|---|---|---|
| run_100 | 54304 | 455.4 | 5 | 13 | 25 | 0.00 |
| run_500 | 417839 | 2320.8 | 5 | 10 | 30 | 0.00 |
| run_1000 | 595605 | 3307.3 | 88 | 160 | 210 | 0.00 |

Point de saturation : entre 500 et 1000 utilisateurs. Au run 1000 le débit ne progresse plus que de
~40 % alors que le P50 passe de 5 à 88 ms et le P99 de 30 à 210 ms : les 5 pods (`maxReplicas`) sont
à ~900m de CPU pour une limite de 1 CPU. RPS max observé : ~3 300 req/s avec 5 pods, sans erreur.

## 3. Scale-up observé

| Run | Pods (max observé) | CPU moyen par pod (kserve-container) | Temps pour atteindre ce nombre de pods |
|---|---|---|---|
| run_100 | 4 | ~130m | non chronométré (quelques dizaines de s) |
| run_500 | 5 (maxReplicas) | ~210m | non chronométré (quelques dizaines de s) |
| run_1000 | 5 (plafond) | ~900m (limite 1 CPU atteinte) | déjà à 5 pods en début de run |

Scale-to-zero : environ 2 min sans trafic (fenêtre 60 s + `scale-down-delay` 30 s + délai de
grâce Knative) · démarrage à froid (`curl -w`) : 3,96 s, puis 0,005 s une fois le pod chaud.

## 4. Analyse canary

- Répartition observée sur 200 requêtes : 177 × 1.0.0 / 23 × 2.0.0 (88,5 % / 11,5 %).
- Cohérent avec 90/10 : 23 au lieu de 20 attendus, dans la variance normale (écart-type ≈ 4,2 sur
  200 tirages). Les adresses dédiées `latest-…` (2.0.0) et `prev-…` (1.0.0) répondent comme prévu.
- Décision : promotion à 100 %. La v2 ne change que le code (route `/version`) avec le même modèle
  v12 et le même seuil ; elle répond correctement sur son adresse dédiée et sur sa part de trafic.
  La révision `00002` reste disponible pour un rollback (`canaryTrafficPercent: 0`).
- Note : faute de connexion rapide, l'image 2.0.0 a été construite par-dessus 1.0.0
  (nouveau `serve.py` + `APP_VERSION`), sans re-télécharger les dépendances.

## 5. Points d'amélioration

- Plusieurs workers uvicorn par pod (ou gunicorn) : le goulot est l'unique worker Python, pas le
  modèle (0,08 ms d'inférence ONNX).
- Augmenter `maxReplicas` ou la limite CPU pour repousser la saturation vue au run 1000.
- Tester la métrique `concurrency` et ajuster `scaleTarget` après mesure du débit réel par pod.
- `minReplicas: 1` aux heures ouvrées pour éviter le démarrage à froid de ~4 s.
- Batching côté client (l'API accepte jusqu'à 1 000 demandes par requête) pour le scoring de masse.
- Automatiser le canary (métriques + rollback automatique) avec GitOps (Lab 6) et le monitoring (Lab 7).

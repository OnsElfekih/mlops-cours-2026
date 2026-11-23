# Rapport — Lab 4 : packaging du modèle

Binôme : Eya & Ons · Dataset : german-credit · Modèle : `scoring-model` v12 (régression logistique, seuil 0,318)

## 1. Comparatif des tailles

| Image | Taille | CVE HIGH + CRITICAL |
|---|---|---|
| `scoring-api:naive` | 1,51 Go | 539 (HIGH 523, CRITICAL 16) |
| `scoring-api:optimized` | 374 Mo | 51 (HIGH 51, CRITICAL 0) |
| `scoring-api:bentoml` | 994 Mo | 94 (HIGH 94, CRITICAL 0) |

## 2. Comparatif des latences

1000 appels par backend après 10 appels de chauffe · x86_64 · Python 3.12.3.

| Backend | P50 (ms) | P95 (ms) | P99 (ms) |
|---|---|---|---|
| sklearn_pipeline | 4.011 | 6.039 | 6.308 |
| onnx_python | 0.080 | 0.085 | 0.089 |
| onnx_batch_100 | 0.345 | 0.357 | 0.374 |

ONNX est ~50× plus rapide que scikit-learn sur une ligne. Test de charge Locust sur l'image
optimisée (100 utilisateurs, 30 s) : 53 494 requêtes, 0 échec, ~1 800 req/s, P50 13 ms,
P95 39 ms, P99 55 ms.

## 3. Interprétation

- **Taille** : l'image optimisée part de `python:3.12-slim` (sans compilateurs ni outils Debian),
  installe les dépendances dans un stage `builder` dont seul `/deps` est copié, et n'embarque que
  le graphe ONNX et onnxruntime : ni scikit-learn, ni MLflow, ni le repo. La naïve copie tout le
  repo (dont `.git`, `data/` et `.env` : fuite de secrets) sur `python:3.12` complet et tourne en root.
  BentoML est entre les deux : il embarque BentoML, scikit-learn et SciPy.
- **Latence** : pour une ligne, le pipeline scikit-learn est dominé par son surcoût fixe en Python
  (validation du DataFrame, sélection des colonnes par nom, un appel par transformateur). ONNX
  Runtime exécute le même graphe compilé en C++ : 0,08 ms. En lot de 100, le coût par ligne tombe
  à ~3,5 µs. En charge, la latence vient surtout de la file d'attente de l'unique worker uvicorn.
- **Scans** : Trivy montre que la base slim élimine ~90 % des CVE (539 → 51) et toutes les
  CRITICAL. ModelScan détecte le pickle piégé (`posix.system`, CRITICAL) : charger un pickle,
  c'est exécuter du code. Le modèle ONNX n'est pas analysé par ModelScan mais il est sûr par
  construction (graphe protobuf déclaratif). Le fichier skops du Registry n'est pas réellement
  analysé non plus (erreurs ModelScan 0.8.8 avec NumPy 2) : sa sécurité repose sur la liste de
  types de confiance de skops (seul `numpy.dtype` est demandé).

## 4. Choix recommandé

`scoring-api:optimized` (FastAPI + ONNX Runtime) : la plus petite (374 Mo), la moins exposée
(51 CVE, aucune CRITICAL), non-root avec HEALTHCHECK, la plus rapide (0,08 ms d'inférence), et
sans pickle. Le modèle est figé dans l'image : une image = une version du modèle, démarrage sans
MLflow, rollback simple. BentoML est plus confortable à développer mais plus lourd et sert un pickle.

## 5. Pistes d'amélioration

- Image distroless (ou `python:3.12-slim` minimal + mises à jour de sécurité) pour réduire encore
  la surface d'attaque et les CVE restantes.
- Retirer pandas des dépendances de l'image (serve.py n'utilise que numpy) : ~45 Mo de moins.
- Quantization / optimisation du graphe ONNX (onnxruntime graph optimizations), utile surtout pour
  des modèles plus gros (forêts, boosting).
- Plusieurs workers uvicorn (ou l'autoscaling du Lab 5) pour absorber la file d'attente en charge.
- Scanner aussi le format skops dès que ModelScan sera compatible NumPy 2, et automatiser
  Trivy + ModelScan dans la CI (Lab 6).

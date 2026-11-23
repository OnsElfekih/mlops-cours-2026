# Rapport — Lab 4 : packaging du modèle

Binôme : Ons ELFEKIH et Eya Guizani
Dataset : `german-credit`
Modèle : `scoring-model` v12
Alias MLflow : `champion`
Seuil de décision : `0.3182`

## 1. Comparatif des tailles

Trois images Docker ont été construites pour comparer leurs tailles, leur fonctionnement et leur sécurité.

| Image | Taille | CVE HIGH + CRITICAL |
|---|---|---|
| `scoring-api:naive` | 2 198 Mo | 502 (487 HIGH, 15 CRITICAL) |
| `scoring-api:optimized` | 429 Mo | 48 (48 HIGH, 0 CRITICAL) |
| `scoring-api:bentoml` | 1 410 Mo | 145 (145 HIGH, 0 CRITICAL) |

L'image naïve repose sur `python:3.12`. Elle copie une grande partie du dépôt et installe les dépendances dans une image contenant de nombreux composants système. Sa taille atteint 2,20 Go.

L'image optimisée repose sur `python:3.12-slim` et utilise une construction Docker multi-stage. Le stage `builder` installe les dépendances, tandis que le stage `runtime` contient les fichiers nécessaires à l'exécution du modèle ONNX.

Les fichiers inutiles, notamment les dossiers de tests et les fichiers Python compilés, ont été supprimés. Un fichier `Dockerfile.dockerignore` limite également les éléments copiés dans l'image.

La taille est passée de 2 197,6 Mo à 428,8 Mo, soit une réduction de 80,5 %.

L'image optimisée utilise un utilisateur non-root (UID 10001) et un HEALTHCHECK Docker fonctionnel.

L'image BentoML atteint 1,41 Go. Elle contient le framework BentoML, le pipeline scikit-learn et les dépendances nécessaires à leur exécution.

Les trois images ont été construites et testées avec succès.

## 2. Comparatif des latences

Le benchmark a été réalisé sous Ubuntu avec Python 3.12, sur le dataset German Credit.

Chaque méthode a été exécutée 1 000 fois, après 10 appels de chauffe.

| Backend | P50 (ms) | P95 (ms) | P99 (ms) |
|---|---|---|---|
| sklearn_pipeline | 3.451 | 4.048 | 4.996 |
| onnx_python | 0.077 | 0.092 | 0.125 |
| onnx_batch_100 | 0.298 | 0.379 | 0.535 |

P50 représente la médiane des temps d'exécution.

P95 indique que 95 % des appels présentent une latence inférieure ou égale à cette valeur.

P99 permet d'étudier les 1 % d'appels les plus lents.

Pour une observation, le pipeline scikit-learn obtient une latence P50 de 3,451 ms, contre 0,077 ms pour ONNX Runtime.

Le facteur d'accélération obtenu est de 44,7.

Pour un lot de 100 observations, ONNX Runtime atteint une latence P50 de 0,298 ms pour le lot complet.

La comparaison des modèles sur les 200 observations du jeu de test a également donné les résultats suivants :

- Écart maximal des probabilités : `8.73e-08`
- Tolérance demandée : `1e-4`
- Décisions identiques : 100 %
- Seuil utilisé : `0.3182`

Le modèle ONNX conserve donc les décisions du modèle scikit-learn sur les données testées, tout en réduisant le temps d'inférence.

Ces mesures concernent l'inférence locale. Elles ne comprennent pas la latence réseau des requêtes HTTP.

## 3. Interprétation

### Pourquoi l'image optimisée est-elle plus petite ?

La réduction de la taille repose sur plusieurs optimisations :

- Utilisation de `python:3.12-slim` au lieu de `python:3.12`.
- Séparation de la construction et de l'exécution avec un Dockerfile multi-stage.
- Installation des dépendances nécessaires uniquement au serving ONNX.
- Exclusion des environnements virtuels, fichiers Git et datasets d'entraînement.
- Suppression des fichiers `.pyc`, des caches Python et des dossiers de tests.
- Copie du modèle ONNX exporté sans les dépendances MLflow et scikit-learn.

Ces modifications ont permis de réduire la taille de l'image de 80,5 %.

### Pourquoi ONNX est-il plus rapide que scikit-learn ?

ONNX représente le modèle sous la forme d'un graphe de calcul pouvant être exécuté par ONNX Runtime.

L'inférence utilise un moteur optimisé et réduit une partie des opérations réalisées dans Python par le pipeline scikit-learn.

Sur notre machine, ONNX Runtime atteint un P50 de 0,077 ms, contre 3,451 ms pour scikit-learn.

Le gain mesuré est de 44,7 fois pour une observation.

ONNX permet également de traiter plusieurs observations dans un seul appel. Le benchmark sur 100 observations donne un P50 de 0,298 ms pour le lot entier.

Ce résultat concerne notre modèle et notre environnement de test. Le gain peut varier selon le modèle, le matériel et la charge.

### Que montrent les scans Trivy et ModelScan ?

Les trois images Docker ont été analysées avec Trivy 0.75.0.

L'image naïve contient 502 vulnérabilités HIGH et CRITICAL détectées, dont 15 CRITICAL.

L'image optimisée contient 48 vulnérabilités HIGH et aucune CRITICAL détectée.

L'image BentoML contient 145 vulnérabilités HIGH et aucune CRITICAL détectée.

L'optimisation Docker a réduit de 90,4 % le nombre de vulnérabilités HIGH et CRITICAL par rapport à l'image naïve.

Les 48 vulnérabilités de l'image optimisée concernent les paquets système Debian détectés par Trivy. Aucun problème HIGH ou CRITICAL n'a été signalé dans les dépendances Python analysées.

ModelScan 0.8.8 a ensuite été utilisé pour analyser les fichiers de modèles.

Le fichier `model.onnx` a été ignoré par ModelScan. Le message `No issues found` ne constitue donc pas une analyse de sécurité complète du modèle ONNX.

Deux fichiers Pickle ont été créés pour démontrer les risques de désérialisation :

- `model.pkl`, contenant le modèle scikit-learn sérialisé.
- `malicious.pkl`, contenant une opération `posix.system`.

ModelScan a détecté une opération dangereuse avec une gravité CRITICAL dans le fichier malveillant.

Le modèle MLflow au format `model.skops` a également été analysé.

Une première exécution a rencontré 11 erreurs de compatibilité entre ModelScan et NumPy.

Un second scan a été réalisé dans un environnement isolé avec NumPy 1.26.4.

Ce scan s'est terminé avec un code retour 0 et sans problème signalé, mais avec huit éléments ignorés.

L'outil skops a identifié le type `numpy.dtype` comme nécessitant une autorisation explicite lors du chargement.

Ces tests montrent l'importance des scans de vulnérabilités, du contrôle des dépendances et de la vérification des formats de sérialisation.

## 4. Choix recommandé

Pour ce projet, l'image recommandée comme candidate au déploiement en production est `scoring-api:optimized`, utilisant FastAPI et ONNX Runtime.

Ce choix repose sur les résultats suivants :

- Taille de 429 Mo, inférieure à l'objectif de 500 Mo.
- Réduction de taille de 80,5 % par rapport à l'image naïve.
- Réduction de 90,4 % des vulnérabilités HIGH et CRITICAL détectées.
- Absence de vulnérabilités CRITICAL détectées dans l'image optimisée.
- Accélération de l'inférence ONNX de 44,7 fois au P50.
- Conservation de 100 % des décisions scikit-learn sur les 200 observations testées.
- Utilisateur non-root avec UID 10001.
- HEALTHCHECK Docker fonctionnel.
- API FastAPI validée avec les endpoints `/health`, `/metadata` et `/predict`.

La validation des requêtes incorrectes a également été vérifiée avec le code HTTP 422.

Les prédictions individuelles et les prédictions par lots ont été testées.

L'image optimisée a été publiée dans le Registry Docker local :

`localhost:5001/scoring-api:1.0.0`

La publication a réussi. Un test de téléchargement avec `docker pull`, suivi de l'exécution du conteneur, a confirmé que l'image publiée fonctionne.

Le digest SHA-256 a été vérifié après le téléchargement.

L'image BentoML constitue une alternative permettant de servir directement le pipeline scikit-learn et de bénéficier des fonctionnalités du framework BentoML.

Cependant, pour notre cas d'usage, l'image FastAPI optimisée présente un meilleur compromis entre taille, latence et nombre de vulnérabilités détectées.

Avant un déploiement en production, les 48 vulnérabilités HIGH restantes devront être examinées et corrigées lorsque des correctifs sont disponibles.

## 5. Pistes d'amélioration
Les améliorations suivantes sont proposées pour la suite du projet. Elles ne sont pas encore réalisées dans le Lab 4.

### 5.1. Automatisation des tests de l'API

Créer un script de test pour vérifier automatiquement les endpoints `/health`, `/metadata` et `/predict` après chaque modification du service FastAPI.

Cette amélioration permettra de détecter les erreurs sans effectuer les tests manuellement.

### 5.2. Mise à jour des dépendances Docker

Mettre à jour les paquets système disposant de correctifs, puis relancer Trivy pour comparer le nombre de vulnérabilités HIGH et CRITICAL.

L'objectif est de réduire les 48 vulnérabilités HIGH détectées dans l'image optimisée.

### 5.3. Ajout d'un journal des prédictions

Enregistrer dans les logs de FastAPI le temps d'exécution, le nombre d'observations traitées et la version du modèle.

Cette amélioration facilitera le suivi des performances et le diagnostic des erreurs.

## 6. Comparaison des travaux d'Ons et d'Eya

Les deux implémentations du Lab 4 ont été comparées avant leur
intégration dans Ons/mlops-improvements.

### 6.1. Résultats des deux environnements

| Indicateur | Ons | Eya |
|---|---:|---:|
| Image Docker optimisée | 428 Mo | 374 Mo |
| ONNX P50 | 0,077 ms | 0,080 ms |
| Accélération ONNX | x44,7 | x49,9 |
| Vulnérabilités HIGH | 48 | 51 |
| Vulnérabilités CRITICAL | 0 | 0 |
| Vérifications Lab 4 | 16/16 | 16/16 |

Les résultats ont été obtenus sur des environnements distincts.
Ils ne constituent pas un benchmark comparatif à matériel identique.

### 6.2. Contributions conservées

Contributions d'Ons :
- API FastAPI et packaging ONNX validés.
- Image Docker multi-stage avec utilisateur non-root.
- Analyse supplémentaire du modèle skops avec NumPy 1.26.
- Rapport détaillé des benchmarks et scans de sécurité.

Contributions d'Eya :
- Script de test de charge packaging/locustfile.py.
- Résultats de charge avec 100 utilisateurs.
- Rapports de benchmark et de sécurité conservés
  dans reports/m4-eya/.
- Documentation des configurations Docker.

Le modèle ONNX conservé est scoring-model v12.
Son seuil de décision est 0.3182.
Son empreinte SHA-256 correspond aux métadonnées.

### 6.3. Validation du code commun

La version intégrée dans Ons/mlops-improvements a été
testée avec les résultats suivants :

- make check-m3 : 20/20.
- make check-m4 TAG=1.0.1 : 16/16.
- Image Docker : localhost:5001/scoring-api:1.0.1.
- Taille mesurée : 428 Mo.
- API FastAPI : /health, /metadata et /predict validés.
- Modèle ONNX : empreinte SHA-256 vérifiée.

### 6.4. Tests de charge Locust

Le test de charge a été exécuté sur l'image commune 1.0.1.

| Indicateur | 5 utilisateurs | 100 utilisateurs |
|---|---:|---:|
| Durée | 15 s | 30 s |
| Requêtes | 1 884 | 40 343 |
| Échecs | 0 | 0 |
| Débit | 130,80 req/s | 1 361,59 req/s |
| P50 | 2 ms | 25 ms |
| P95 | 3 ms | 62 ms |
| P99 | 5 ms | 84 ms |

Le scénario de 100 utilisateurs a été exécuté avec une
vitesse de démarrage de 10 utilisateurs par seconde.

Locust a signalé une utilisation CPU supérieure à 90 %.
Cette saturation peut influencer le débit et les latences.

Eya avait mesuré 53 494 requêtes sans échec sur son
environnement, avec un P50 de 13 ms et un P95 de 39 ms.

Les deux environnements ont donc validé le traitement
des requêtes sans erreur dans leurs scénarios respectifs.

Les fichiers CSV du test commun sont conservés dans reports/.

### 6.5. Bilan de l'intégration

La fusion conserve les contributions des deux membres.
Les contrôles des Labs 3 et 4 sont réussis.

L'image Docker commune a été reconstruite, publiée dans
le Registry local et validée avec le script Locust d'Eya.

Les rapports individuels restent disponibles pour
assurer la traçabilité des résultats.

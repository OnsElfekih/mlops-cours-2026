# Rapport comparatif — Kubeflow Pipelines vs Airflow (Lab 2)


**Binôme :** ELFEKIHOns/GUIZANIEya · **Date :** 22Octobre2026 · **Machine :** ASUS VivoBook X515EP — Intel Core i5-1135G7,
16 Go RAM, Ubuntu Linux ·
**Dataset :** german-credit

## 1. Contexte

Ce laboratoire consiste à développer et comparer deux solutions d'orchestration de pipelines Machine Learning : Kubeflow Pipelines (KFP) et Apache Airflow. Le projet utilise le dataset German Credit pour prédire le risque de crédit à partir de caractéristiques financières et personnelles. Le pipeline comprend six étapes : téléchargement des données, validation, préparation, entraînement, évaluation et promotion du modèle. Les deux orchestrateurs exécutent les composants dans des conteneurs Kubernetes à partir de la même image Docker (`localhost:5001/scoring-pipeline:1.0.0`). Les artefacts sont stockés dans SeaweedFS via une interface compatible S3, tandis que MLflow assure le suivi des expériences, des métriques et des versions du modèle. Cette architecture permet de comparer les orchestrateurs en conservant les mêmes traitements et la même infrastructure.

## 2. Résultats quantitatifs

| Mesure | KFP | Airflow |
|---|---|---|
| Installation de zéro (min) | Non mesuré | Non mesuré  |
| Temps jusqu'au premier run réussi (min) | Non mesuré  | Non mesuré  |
| Durée d'un run complet (min) | 2,17 (test cache 1) | 0,83 |
| Relance avec les mêmes paramètres (min) | 2,18 (test cache 2) | 0,80 |
| Hot-reload : modifier un step et voir le résultat (min) | Non mesuré | Non mesuré |
| Lignes de code propres à l'orchestrateur | 294 | 64 |
| Mémoire consommée par l'outil (Go, `docker stats`) | ≈ 3,01 Gio (cluster Kind partagé) | ≈ 1,79 Gio (stack Docker Compose) |

## 3. Analyse qualitative

### Developer experience

KFP permet de définir les composants et leurs dépendances en Python, puis de compiler et d'exécuter le pipeline sur Kubernetes. Son interface graphique facilite la compréhension des étapes et des artefacts. Airflow utilise un DAG Python plus compact dans notre implémentation, avec des tâches KubernetesPodOperator. La modification du DAG est relativement simple, mais sa prise en compte nécessite parfois une actualisation par Airflow.

### Scaling

Les deux orchestrateurs peuvent exécuter des tâches dans des pods Kubernetes. KFP est particulièrement adapté aux workflows ML composés de plusieurs étapes avec des artefacts et des dépendances explicites. Airflow convient également à l'orchestration de traitements variés, notamment les workflows combinant ingestion, transformation et Machine Learning. Aucun test de montée en charge n'a été réalisé dans ce laboratoire.

### Debugging

KFP fournit un graphe des composants, leurs statuts et les artefacts produits. Le deuxième test a confirmé la réutilisation du cache pour l'ingestion, la validation et la préparation, tandis que les étapes suivantes ont été réexécutées. Airflow facilite l'inspection des tâches et de leurs journaux. Nous avons notamment diagnostiqué un problème de traçabilité MLflow, corrigé en transmettant la variable LAUNCHED_BY=airflow.

## 4. Recommandation

Pour un projet principalement orienté Machine Learning, nous recommandons KFP lorsque la traçabilité des composants, la gestion des artefacts et le caching sont prioritaires. Airflow est préférable pour orchestrer des workflows plus généraux, combinant traitements de données, automatisations et tâches ML. Le choix dépend donc des besoins du projet, des compétences de l'équipe et de l'infrastructure disponible.

## 5. Limites de la comparaison

Cette comparaison repose sur un seul dataset, un modèle simple et une machine locale aux ressources limitées. Les temps d'installation et de développement n'ont pas été chronométrés. Les mesures mémoire couvrent des périmètres différents : le cluster Kind partagé pour KFP et les services Docker Compose pour Airflow. Les durées ne correspondent pas non plus à des conditions identiques de cache et de version. Il serait nécessaire de réaliser plusieurs exécutions contrôlées, avec une charge identique, pour comparer rigoureusement les performances.

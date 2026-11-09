# Rapport comparatif — KFP vs Airflow (Lab 2)

**Binôme :** Non renseigné · **Date :** 09/11/2026 · **Machine :** 16 CPU, 15 GiB RAM, Ubuntu 24.04.4 · **Dataset :** german-credit

## 1. Contexte

Ce laboratoire consiste à créer un pipeline ML de six étapes : téléchargement, validation, préparation, entraînement, évaluation et enregistrement du modèle. Il utilise le dataset German Credit (1 000 lignes). Les deux orchestrateurs exécutent les mêmes étapes avec la même image Docker sur un cluster Kubernetes local (Kind).

## 2. Résultats quantitatifs

| Mesure | KFP | Airflow |
|---|---|---|
| Installation (min) | Non mesuré | Non mesuré |
| Durée d'un run (min) | ~2 | ~0,8 |
| Relance / Hot-reload | Non mesuré | Non mesuré |
| Lignes de code | 273 | 69 |
| Mémoire (Go) | Non disponible | ~1,83 |

La mémoire d'Airflow est issue de docker stats. La mesure de KFP était indisponible.

## 3. Analyse qualitative

**Developer experience :** KFP facilite la création de pipelines ML et le suivi des artefacts. Airflow utilise des DAGs Python plus compacts, mais nécessite S3 et XCom pour partager les données.

**Scaling :** Les deux outils peuvent exécuter des tâches en parallèle sur Kubernetes. KFP est orienté ML, tandis qu'Airflow convient à différents types de workflows.

**Debugging :** Les deux interfaces permettent de suivre les tâches et de consulter leurs logs pour identifier les erreurs.

## 4. Recommandation

KFP est adapté aux projets ML qui nécessitent le suivi des expériences et des modèles. Airflow est préférable pour les workflows généraux qui combinent plusieurs traitements et sources de données.

## 5. Limites de la comparaison

Le laboratoire utilise un environnement local. Le scaling en production, la haute disponibilité et les performances sur de grands volumes de données n'ont pas été testés. Certaines mesures, comme les temps d'installation, ne sont pas disponibles.

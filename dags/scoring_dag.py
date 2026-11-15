
"""Pipeline de scoring German Credit orchestré par Airflow avec XCom."""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes import client as k8s


IMAGE = "localhost:5001/scoring-pipeline:1.4.0"

WORKSPACE = (
    "s3://airflow-workspace/"
    "{{ dag_run.run_id | replace(':', '_') | replace('+', '_') }}"
)

ENV_VARS = {
    "AWS_ENDPOINT_URL": "http://s3.storage.svc.cluster.local:8333",
    "AWS_DEFAULT_REGION": "us-east-1",
    "MLFLOW_TRACKING_URI": "http://mlflow.mlflow.svc.cluster.local:5000",
    "LAUNCHED_BY": "airflow",
}

S3_SECRET = [
    k8s.V1EnvFromSource(
        secret_ref=k8s.V1SecretEnvSource(name="s3-credentials")
    )
]


def create_task(
    task_id,
    module,
    *args,
    xcom=False,
    memory="1Gi",
    retries=0,
):
    """Crée une tâche Kubernetes avec partage S3 et XCom optionnel."""

    return KubernetesPodOperator(
        task_id=task_id,
        name=task_id.replace("_", "-"),
        namespace="airflow",
        image=IMAGE,
        cmds=["python", "-m", "pipelines.stage"],
        arguments=[
            module,
            "--workspace",
            WORKSPACE,
            *(["--xcom"] if xcom else []),
            "--",
            *args,
        ],
        env_vars=ENV_VARS,
        env_from=S3_SECRET,
        kubernetes_conn_id="kubernetes_default",
        image_pull_policy="Always",
        container_resources=k8s.V1ResourceRequirements(
            requests={
                "cpu": "250m",
                "memory": "512Mi",
            },
            limits={
                "memory": memory,
            },
        ),
        retries=retries,
        get_logs=True,
        on_finish_action="delete_pod",
        in_cluster=False,
        do_xcom_push=xcom,
    )


with DAG(
    dag_id="scoring_dag",
    description="Pipeline German Credit avec Kubernetes, MLflow et XCom",
    start_date=datetime(2026, 10, 1),
    schedule=None,
    catchup=False,
    default_args={
        "owner": "mlops-student",
        "retry_delay": timedelta(seconds=30),
    },
    tags=["mlops", "scoring", "kubernetes", "xcom"],
) as dag:

    # Étape 1 : téléchargement des données
    download = create_task(
        "download",
        "src.download",
        retries=3,
    )

    # Étape 2 : validation des données
    validate = create_task(
        "validate",
        "src.validate_data",
        memory="512Mi",
    )

    # Étape 3 : préparation des données
    prepare = create_task(
        "prepare",
        "src.prepare",
    )

    # Étape 4 : entraînement et publication du run_id MLflow
    train = create_task(
        "train",
        "src.train",
        xcom=True,
        memory="2Gi",
    )

    # Étape 5 : évaluation et publication des métriques
    evaluate = create_task(
        "evaluate",
        "src.evaluate",
        xcom=True,
    )

    # Étape 6 : promotion du modèle entraîné par ce DAG
    promote = create_task(
        "promote",
        "src.promote",
        "--run-id",
        "{{ ti.xcom_pull(task_ids='train')['mlflow_run_id'] }}",
    )

    download >> validate >> prepare >> train >> evaluate >> promote

"""Pipeline de scoring German Credit orchestré par Airflow."""

from datetime import datetime

from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes import client as k8s

IMAGE = "localhost:5001/scoring-pipeline:1.0.0"

WORKSPACE = "s3://airflow-workspace/{{ dag_run.run_id | replace(':', '_') | replace('+', '_') }}"

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

with DAG(
    dag_id="scoring_dag",
    description="Pipeline German Credit avec Kubernetes et MLflow",
    start_date=datetime(2026, 10, 1),
    schedule=None,
    catchup=False,
    tags=["mlops", "scoring", "kubernetes"],
) as dag:

    def create_task(task_id, module):
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
            ],
            env_vars=ENV_VARS,
            env_from=S3_SECRET,
            kubernetes_conn_id="kubernetes_default",
            image_pull_policy="Always",
            get_logs=True,
            on_finish_action="delete_pod",
            in_cluster=False,
            do_xcom_push=False,
        )

    download = create_task("download", "src.download")
    validate = create_task("validate", "src.validate_data")
    prepare = create_task("prepare", "src.prepare")
    train = create_task("train", "src.train")
    evaluate = create_task("evaluate", "src.evaluate")
    promote = create_task("promote", "src.promote")

    download >> validate >> prepare >> train >> evaluate >> promote

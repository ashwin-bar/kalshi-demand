"""MLflow tracking on the USB drive (D:), with retries for transient USB I/O errors (row 106)."""
import time
import mlflow
from kalshi_api import DATA_ROOT

DB = DATA_ROOT / "mlflow.db"
URI = f"sqlite:///{DB.as_posix()}"
EXPERIMENT = "kalshi-demand-v1"


def setup(experiment=EXPERIMENT, attempts=5):
    """Point MLflow at the database on D: and select the experiment, retrying if the drive is briefly unavailable."""
    for i in range(1, attempts + 1):
        try:
            mlflow.set_tracking_uri(URI)
            if mlflow.get_experiment_by_name(experiment) is None:
                mlflow.create_experiment(experiment, artifact_location=(DATA_ROOT / "mlartifacts").as_uri())
            mlflow.set_experiment(experiment)
            return
        except Exception as e:
            print(f"  MLflow setup failed ({type(e).__name__}); retry {i}/{attempts} in 5s")
            time.sleep(5)
    raise RuntimeError("MLflow tracking database on D: is unavailable")


def log_run(name, params, metrics, tags, attempts=5):
    """Log one run (params, metrics, tags), retrying on transient errors."""
    for i in range(1, attempts + 1):
        try:
            with mlflow.start_run(run_name=name):
                mlflow.set_tags(tags)
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
            return
        except Exception as e:
            print(f"  logging '{name}' failed ({type(e).__name__}); retry {i}/{attempts} in 5s")
            time.sleep(5)
    raise RuntimeError(f"Could not log run '{name}' after {attempts} attempts")
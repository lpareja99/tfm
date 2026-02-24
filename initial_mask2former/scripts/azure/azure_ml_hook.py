# scripts/azure/azure_ml_hook.py
from mmengine.hooks import Hook
from azureml.core import Run

class AzureMLLogHook(Hook):
    """Log metrics to Azure ML at the end of validation."""

    def __init__(self):
        super().__init__()
        # Get the current Azure ML run (works inside a job)
        self.run = Run.get_context()

    def after_val_epoch(self, runner, metrics=None):
        """After a validation epoch, log the metrics to Azure ML."""
        if metrics is None:
            metrics = {}
        # Log each metric
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.run.log(key, value)
            elif isinstance(value, dict):
                # For per-class metrics, log as a table or individual values
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        self.run.log(f"{key}_{sub_key}", sub_value)
        # Optionally log the full metrics dict as a JSON
        self.run.log_row("validation_metrics", **metrics)
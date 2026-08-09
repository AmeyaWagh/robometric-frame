# FRAME Examples

This directory contains example scripts demonstrating the usage of robometric-frame library.

## Examples

### 1. Basic Success Rate (`basic_success_rate.py`)

Demonstrates fundamental usage of the SuccessRate metric:
- Binary success indicators
- Multiple batch updates
- Continuous scores with threshold
- Real-world scenario (pick-up tasks)

**Run:**
```bash
python examples/basic_success_rate.py
```

### 2. Distributed Training (`distributed_training.py`)

Shows how to use SuccessRate in distributed training contexts:
- Multi-GPU evaluation simulation
- Batch evaluation in training loops
- Metric synchronization across processes

**Run:**
```bash
python examples/distributed_training.py
```

## Integration Examples

### PyTorch Training Loop

```python
import torch
from robometric_frame import SuccessRate

# Initialize metric
success_metric = SuccessRate()

for epoch in range(num_epochs):
    for batch in dataloader:
        # Your model inference
        predictions = model(batch)

        # Evaluate task success
        success = evaluate_tasks(predictions, batch.targets)

        # Update metric
        success_metric.update(success)

    # Compute epoch success rate
    epoch_sr = success_metric.compute()
    print(f"Epoch {epoch} Success Rate: {epoch_sr:.2%}")

    # Reset for next epoch if needed
    success_metric.reset()
```

### PyTorch Lightning

```python
import pytorch_lightning as pl
from robometric_frame import SuccessRate

class RobotPolicyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.success_rate = SuccessRate()

    def validation_step(self, batch, batch_idx):
        predictions = self(batch)
        success = self.evaluate_success(predictions, batch)
        self.success_rate.update(success)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        sr = self.success_rate.compute()
        self.log("val_success_rate", sr)
        self.success_rate.reset()
```

### Hugging Face Transformers

```python
from transformers import Trainer
from robometric_frame import SuccessRate

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    success = evaluate_task_success(predictions, labels)

    metric = SuccessRate()
    metric.update(torch.tensor(success))

    return {
        "success_rate": metric.compute().item()
    }

trainer = Trainer(
    model=model,
    compute_metrics=compute_metrics,
    # ... other arguments
)
```

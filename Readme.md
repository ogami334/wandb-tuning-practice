# Run single experiment with hyperparameter modification
```bash
python train.py --

# Tune hyperparameters using wandb sweep
```bash
wandb sweep sweep_sgd.yaml
```
if you run the above command, you will get a sweep id. You can use this id to run the sweep agent.

```bash
wandb agent --count 1 <sweep_id>
```
You can specify the number of runs each agent should run by using the `--count` argument.

# Reference
- [PyTorch Lightning Tutorial](https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/cifar10-baseline.html)

# wandb-tuning-practice 

## Run single experiment with hyperparameter modification
```bash
python train.py optimizer.lr=0.001
```

## Tune hyperparameters using wandb sweep
```bash
wandb sweep sweep_sgd.yaml
```
if you run the above command, you will get a sweep id. You can use this id to run the sweep agent.

```bash
wandb agent --count 1 <sweep_id>
```
You can specify the number of runs each agent should run by using the `--count` argument.

## Reference
- [PyTorch Lightning CIFAR10 Tutorial](https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/cifar10-baseline.html)
- [Wandb Sweep using hydra](https://wandb.ai/adrishd/hydra-example/reports/Configuring-W-B-Projects-with-Hydra--VmlldzoxNTA2MzQw?galleryTag=posts&utm_source=fully_connected&utm_medium=blog&utm_campaign=hydra)

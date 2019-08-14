# Examples

### [Improved Wasserstein Generative Adversarial Network](improved_wgan.py)

An implementation of the "gradient penalty" method for training Wasserstein-GANs described [in this paper](https://arxiv.org/abs/1704.00028)

Demonstrates how to specify configurations, create simple sequential models, create multiple trainers, visualize progress, and create/restore snapshots.

-   Run `python examples/improved_wgan.py` to start training
-   To visualize training progress and the learned distribution, run: `tensorboard --logdir /tmp/wgan-gp/summary/`

### [Linear Regression](linear_regression.py)

Obligatory simple linear regression example.

---

For more elaborate models, see [Alchemy](https://github.com/ethereon/alchemy).

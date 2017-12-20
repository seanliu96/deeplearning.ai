## Tuning Process

### Hyperparameters

$$
\alpha: \text{learning rate (most important)} \\
\beta_1 \approx 0.9: \beta\text{ in motentum} \\
\beta_2 \approx 0.999: \beta\text{ in RMSprop} \\
n^{[l]}: \text{# of hidden units} \\
\text{mini-batch size} \\
L: \text{# of layers} \\
\text{learning rate decay} \\
$$

Try random values and do not use a grid

Coarse to fine search

## Using an Appropriate Scale to Pick hyperparameters

$$
n^{[l]}: \text{# of hidden units (uniform scale)} \\
L: \text{# of layers (uniform scale)} \\
\alpha: \text{learning rate  (log scale)} \\
\beta: \text{(log scale)}
$$

## Hyperparameters Tuning in Practice: Pandas vs. Caviar

Intuitions do get stale. Re-evaluate occasionally.

### Babysitting One Model

If you have maybe a huge data set but not a lot of computational resources, not a lot of CPUs and GPUs, so you can basically afford to train only one model or a very small number of models at a time. In that case you might gradually babysit that model even as it's training.  People that babysit one model, that is watching performance and patiently nudging the **learning rate** up or down. But that's usually what happens if you don't have enough computational capacity to train a lot of models at the same time.

### Training Many Models in Parallel

You might train many different models in parallel, where these orange lines are different models, right, and so this way you can try a lot of different hyperparameter settings and then just maybe quickly at the end pick the one that works best. Looks like in this example it was, maybe this curve that look best.
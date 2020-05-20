This repo contains code for training and evaluating populations of agents
that learn to communicate to solve a simple reference game.
It implements the experiments from the paper [Emergence of Compositional Language with Deep Generational Transmission](https://arxiv.org/abs/1904.09067)
and is built on top of the [code](https://github.com/batra-mlp-lab/lang-emerge) for the paper 
[Natural Language Does Not Emerge 'Naturally' in Multi-Agent Dialog](https://arxiv.org/abs/1706.08502).

If you find this code useful, please consider citing the original work:

```
@misc{cogswell2019emergence,
    title={Emergence of Compositional Language with Deep Generational Transmission},
    author={Michael Cogswell and Jiasen Lu and Stefan Lee and Devi Parikh and Dhruv Batra},
    year={2019},
    eprint={1904.09067},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```


Setup and Requirements
===

This repository is compatible with Python 3.7 and PyTorch 0.4.1.
Details about all dependencies can be found in conda_env.yml.
In particular, a conda environment with all the needed dependencies can be
installed by running

```sh
$ conda env create -f conda_env.yml
```


Running the Code
===

The workflow is to train models with `run.py -m train`, evaluate
them with `run.py -m evaluate`, then analyze and visualize the results
in the `analysis.ipynb` notebook. The training and evaluation scripts
are described in more detail below while the analysis notebook has
explanations throughout.


Training
---

Our models are trained using the `run.py` script, which decodes the
configuration details for a particular experiment code then uses
that configuration to call `train.py`:

```
./run.py -m train exp7.<model>.<replacement>.<dataset>.<split>.<seed>
```

Model architecture (`<model>`):
* 0 - Memoryless + Minimal Vocab
* 1 - Overcomplete
* 2 - Minimal Vocab
* 3 - Memoryless + Overcomplete

Replacement method (`<replacement>`):
* 0 - Single No Replacement
* 5 - Single Random
* 6 - Single Oldest
* 1 - Multi No Replacement
* 11 - Multi Uniform Random
* 12 - Multi Epsilon Greedy
* 10 - Multi Oldest

Dataset (`<dataset>`):
* 0 - Our dataset w/ cross-val
* 3 - Kottur et al w/out cross-val

Each value of `<seed>` is a value from 0 to 3 (inclusive) that specifies the random seed used for initialization.

Each value of `<split>` is cross val fold from 0 to 3 (inclusive) that specifies the fold (of 4 folds) used for evaluation.

See L326 of `run.py` to see how these are decoded to arguments of train.py. Also see `run.py -h` for additional options that might help with gpu usage and training on a slurm cluster.


Evaluation
---

This script evaluates a specified model which has already been trained
and saves the results to a file for later analysis in `analysis.ipynb`.

```
./run.py -m evaluate <expcode>
```

`<expcode>`: This essentially specifies the model to evaluate; it's a
code that looks something like `exp7.<model>.<replacement>.<dataset>.<split>.<seed>` from the [Training](#training) section.

The above command launches `evaluate.py` with different arguments
depending on what you want to be valuated.
For a basic evaluation of test accuracy see the commands at L1025
and L1031 of `run.py` (uncomment the corresponding lines before running).
To generate data for the language comparison analysis see L1037.
To generate data for the language visualizations see L1044.

Tools
---

[visdom](https://github.com/facebookresearch/visdom) is used to monitor training.
The `plot_logs.py` script parses training logs automatically saved to 
`data/experiments/` and posts them to a visdom server specified by the
`.visdom_config.json` file (see an example in `.visdom_config.json.example`).

If running on a slurm cluster, running the `slurm_jobs.py` script
lists all running jobs by reading from recent logs in `data/logs/`.


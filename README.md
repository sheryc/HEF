# The Hierarchy Expansion Framework (HEF)

This repo contains source code of [Enquire One’s Parent and Child Before Decision: Fully Exploit Hierarchical Structure for Self-Supervised Taxonomy Expansion](https://arxiv.org/pdf/2101.11268.pdf), published at The Web Conference 2021.

## Installation Guide

Nvidia's apex library is needed. Please follow the [official installation guide of apex](https://github.com/NVIDIA/apex#quick-start) to begin.

The model was natively developed to support torch's Distributed Data Parallel (DDP) training using multiple GPUs, and offer full support for multi-gpu training and inference. Since single-GPU training is not natively supported during development, if you find any bugs, please submit an issue.

The current implementation does not support CPU training, since large pretrained models need to be finetuned during training.

The model was developed and tested with `torch==1.4.0` and `networkx==2.4`. For other requirements, please refer to `requirements.txt` and install the required libraries with `pip install -r requirements.txt`.

## Data Preparation

### Datasets used in the paper 

For dataset used in our paper, you can directly download all input files below and skip this section. For dataset file named `<TAXONOMY_NAME>.bert.pickle.bin`, create a folder in `./data/` named `<TAXONOMY_NAME>`  and put this file into the created folder.

* SemEval16-Env: https://t.ly/ZVlY
* SemEval16-Sci: https://t.ly/B0QT
* SemEval16-Food: https://t.ly/HqPM

### Other datasets

#### Step 1: Organize the input taxonomy

First, create a folder in `./data/` named `<TAXONOMY_NAME>`.  `<TAXONOMY_NAME>` is the name of the taxonomy, and should match the following file names. Then, create the following two files:

1. `<TAXONOMY_NAME>.terms`. Each line represents one concept in the taxonomy, including its ID and surface name.

```
taxon1_id \t taxon1_surface_name
taxon2_id \t taxon2_surface_name
taxon3_id \t taxon3_surface_name
...
```

2. `<TAXONOMY_NAME>.taxo`. Each line represents one edge in the taxonomy, including the parent taxon ID and child taxon ID.

```
parent_taxon1_id \t child_taxon1_id
parent_taxon2_id \t child_taxon2_id
parent_taxon3_id \t child_taxon3_id
...
```
Notes: Make sure the `<TAXONOMY_NAME>` is the same in the 2 files and the dataset folder name.

#### Step 2 (Optional): Generate train/validation/test partition files

You can generate your desired train/validation/test parition files by creating another 3 separated files (named `<TAXONOMY_NAME>.terms.train`, `<TAXONOMY_NAME>.terms.validation`, and `<TAXONOMY_NAME>.terms.test`) and puting them in the same directory as the above two required files.

These three partition files are of the same format -- each line includes one `taxon_id` that appears in the above `<TAXONOMY_NAME>.terms` file. Make sure that the validation and test nodes are leaf nodes.

Notice: if train/validation/test partition are used, set `config['dataset']['existing_partition']` to `False`, or the `Dataset` class will automatically take 80% nodes for training set, 20% nodes for test set (all leaf nodes), and 10 test nodes will be separated to be validation set. The separation results will be saved as `<TAXONOMY_NAME>.bert.pickle.bin` for fast loading in the same folder. This `<TAXONOMY_NAME>.bert.pickle.bin` file is the only data requirement for the model. As long as this file exists, the dataset splits would remain unchanged.

## Training

### Quick start

For the 3 benchmark datasets in the paper, we provide 3 config files containing all parameter settings in `./config_files/` for each dataset. Directly training with these three configs can generate results similar to our reported results in the paper, with seed fixed as 42.

For single-card training, a config file need to be specified. For instance, if device 0 is used:

```
python train.py --config config_files/config.se16_env.json --device 0
```

or:

```
CUDA_VISIBLE_DEVICES=0 python train.py --config config_files/config.se16_env.json
```

For multi-GPU training, you need to specify the GPU counts for DDP. For instance, if device 0 and 1 are used:

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --config config_files/config.se16_env.json
```

### Specifying hyper-parameters

You can change the hyper-parameters in each settings. There are two ways to achieve this.

1. Directly modify the config file. All changable hyper-parameters and settings are specified in the config file.
2. Add arguments to the training script. Currently, you can change training epochs, multi-task learning weight $\eta$ and learning rate by adding ``--epochs``, ``-eta`` and ``--lr``, repectively, after the training script. For example:

````
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --config config_files/config.se16_env.json --eta 0.5 --lr 1e-4
````

You can also add other shortcuts to modify other hyper-parameters. Follow the construction of variable ``options`` in ``train.py``'s  entrance to build other shortcuts for hyper-parameter modification.

If the current setting is hard to train on your GPU, open the config file, try to reduce ``node_batch_size`` in ``train_dataloader`` and in ``test_dataloader``, and raise ``accumulation_steps`` in ``trainer``. This might result in more training time, but less memory usage and similar result.

Note that multi-GPU training and single-GPU training end up with different results due to the mini-batch splitting in DDP.

## Testing

For testing, all you need to do is to specify a model checkpoint, and all other configurations will be automatically set according to the config and arguments for training the same model. Testing also supports multi-GPU acceleration. During training, all the checkpoints and configs are saved under ``./.saved/``, and the checkpoint with best performance on validation set is named as ``model_best.pth``.

For single-card testing, a checkpoint need to be specified. For instance, if device 0 is used:

````
python test.py --resume .saved/models/HEF_env/release/model_best.pth --device 0
````

or:

````
CUDA_VISIBLE_DEVICES=0 python test.py --resume .saved/models/HEF_env/release/model_best.pth
````

For multi-GPU testing, you need to specify the GPU counts for DDP. For instance, if device 0 and 1 are used:

````
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 test.py --resume .saved/models/HEF_env/release/model_best.pth
````

You can run case study by adding ``--case_study`` argument, which will generate a file with 5 hit cases, 5 worst cases and 5 random cases in ``./case_studies/``, with detailed analysis of both the selected parent and the ground-truth parent's: 1. anchor's parent's Forward Score, 2. anchor's Pathfinder Score and 3. Current Score, and 4. the anchor's child with max Pathfinder Score's Backward Score.

Similar as training, if the current setting is hard to test on your GPU, open the config file, try to reduce ``node_batch_size`` in ``test_dataloader``.

Multi-GPU testing/validation and single-GPU testing/validation would generate the same results.

## Pretrained Models - TO BE RELEASED SOON

Due to the training cost for each model, we release the pretained models for the 3 datasets. These models are the resulting ``model_best.pth`` trained by the default settings in the given config files. Put the downloaded `.saved/` folder in the root path of this repo. 

The offered pretrained models are trained on 2 GPUs for SemEval16-Env and SemEval16-Sci, and 4 GPUs for SemEval16-Food.

## Some Notes About Hyper-parameter Selection

The hyper-parameters we used to obtain the reported results in the paper has some difference with the configs in this repo, since we tuned the hyper-parameters using BO to obtain the reported results. The configs in this repo accords with the experiment settings stated in the paper (except for SemEval16-Food since the current implementation is not efficient enough), thus, the obtained results might be a bit lower in some metrics while a bit higher in the others than the results in the paper. However, the minor change in evaluation performance due to hyper-parameter selection does not affect the overall performance of HEF, since it can still surpass the baselines by a large margin.


## Model Organization

For our own HEF implementation, we follow the project organization in [TaxoExpan](https://github.com/mickeystroller/TaxoExpan) and [pytorch-template](https://github.com/victoresque/pytorch-template).


# Pixle: a fast and effective black-box attack based on rearranging pixels

This repository contains a PyTorch implementation of the paper: 

### [Pixle: a fast and effective black-box attack based on rearranging pixels](https://arxiv.org/abs/2105.02551)
<!--[Pixle: a fast and effective black-box attack based on rearranging pixels](https://arxiv.org/abs/2105.02551)\ -->
[Jary Pomponi](https://www.semanticscholar.org/author/Jary-Pomponi/1387980523), [Simone Scardapane](https://www.sscardapane.it/), [Aurelio Uncini](http://www.uncini.com/)

### Abstract
Recent research has found that neural networks are vulnerable to several types of adversarial attacks, where the input samples are modified in such a way that the model produces a wrong prediction that misclassifies the adversarial sample. In this paper we focus on black-box adversarial attacks, that can be performed without knowing the inner structure of the attacked model, nor the training procedure, and we propose a novel attack that is capable of correctly attacking a high percentage of samples by rearranging a small number of pixels within the attacked image. We demonstrate that our attack works on a large number of datasets and models, that it requires a small number of iterations, and that the distance between the original sample and the adversarial one is negligible to the human eye. 

### Main Dependencies
* pytorch==1.7.1
* python=3.8.5
* torchvision==0.8.2
* pyyaml==5.3.1
* tqdm

### Experiments files
The folder './configs/' contains all the yaml files used for the experiments presented in the paper. 

The folder './config/attacks' contains the files containing all the attacks with the respective hyperparameters. 

### Attacks implementation

The attack can be found in the file attacks/psa.py 

### Training
The only training file is main.py. 

So see how to use it to lunch the experiments, please refer to the files:
* experiments.sh
* experiments_targeted.sh
* experiments_dimensions.sh

All the above files take as input the dataset, the architecture and the device to be used, with some limitations. 
Please refer to each file to understand how to launch it.

### Cite

Please cite our work if you find it useful:

```
@misc{pomponi2021structured,
      title={Structured Ensembles: an Approach to Reduce the Memory Footprint of Ensemble Methods}, 
      author={Jary Pomponi and Simone Scardapane and Aurelio Uncini},
      year={2021},
      eprint={2105.02551},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
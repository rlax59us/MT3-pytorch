# MT3-pytorch for MAESTRO dataset

Now, this is an unofficial implementation of [SEQUENCE-TO-SEQUENCE PIANO TRANSCRIPTION WITH TRANSFORMERS](https://arxiv.org/abs/2107.09142v1) in pytorch.

Converted original model code in [MT3 repository](https://github.com/magenta/mt3/tree/main) from jax to pytorch.

Later, I will update to extend the model to multi-track and multi-task for implementing the MT3 model with [Slakh2100 dataset](http://www.slakh.com/).

## Prerequisite
First of all, please install the appropriate version of pytorch library.

With Anaconda, you can install using below command line(example).
```
$ conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch 
```

After then, install the specified libraries in requirements.txt file.
```
$ pip install -r requirements.txt
```

## Usage

Not done yet.

### Train
```
$ python train.py
```


## Results


## Citations

```bibtex
@article{
  title={SEQUENCE-TO-SEQUENCE PIANO TRANSCRIPTION WITH TRANSFORMERS},
  author={Curtis Hawthorne, Ian Simon, Rigel Swavely, Ethan Manilow and Jesse Engel},
  paper={https://arxiv.org/abs/2107.09142v1},
  year={2021}
}
```

# FeatureNorm: L2 Feature Normalization for Dynamic Graph Embedding (ICDM2020)

This repository is an implementation of our ICDM 2020 paper [FeatureNorm: L2 Feature Normalization for Dynamic Graph Embedding](https://arxiv.org/abs/2103.00164)

## 1. Main idea
Illustration of learning process without normalization         |
:-------------------------:|
![](https://i.imgur.com/2kDSxgN.png)

Illustration of learning process with the proposed normalization            |
:-------------------------:|
![](https://i.imgur.com/o44aSqJ.png)

Illustration of the main idea for L2 feature normalization. Each sub-figure illustrates the node embeddings of the corresponding time step. The upper three sub-figures demonstrate that the unconstrained node embeddings, including all nodes, are getting closer and closer along with time steps. The lower three sub-figures show that the node embeddings with the proposed normalization constraint makes connected (or similar) nodes gather in the adjacent area while pushes unconnected (or dissimilar) nodes far away.

## 2. Usage
Our core code is quiet simple, in the following, we give a short example to show how to use it:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import args


class Net(nn.Module):
  def __init__(self, args):
    super(Net, self).__init__()
    self.model = DynGCN(args)

  def forward(self, timestamp, inputs):
    # to obtain the representation of each step
    embeddings = []
    for t in timestamp:
      x = self.model(inputs)

      if normalize == 'fn':
        x = x - x.mean(dim=0) # centering
        x = F.normalize(x) # normalization
      if normalize == 'no':
        x = x

      embeddings.append(x)
      return embeddings

```
Note that the normalization is the final step, and there is no other operations (e.g, gcn layer, sigmoid, relu, leakly relu, mlp, etc) on the embedding. For more example, please see other folders for deatils.

## 3. Attribution

If you use this code or our results in your research, please cite:
```
@article{Marlin2020Featurenorm,
  author    = {Menglin Yang and
               Ziqiao Meng and
               Irwin King},
  title     = {FeatureNorm: {L2} Feature Normalization for Dynamic Graph Embedding},
  journal   = {Arxiv},
  volume    = {abs/2103.00164},
  year      = {2021},
  url       = {https://arxiv.org/abs/2103.00164},
}
```

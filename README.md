# API for SuctionNet-1Billion

API for SuctionNet-1Billion dataset of RA-L paper "SuctionNet-1Billion:  A  Large-Scale  Benchmark  for  Suction  Grasping" 

## Dataset

Download data and labels from our [SuctionNet webpage](https://graspnet.net/suction).

## Suction Definition

A suction is defined by its 3D suction point and direction. The direction is a normalized vector pointing outwards the object surface. See the image below.

To evaluate your algorithm, you should represent your predicted suction as a 7-dimensional vector. The first element is your predicted score, the following three elements are 3D suction point coordinate and the last three are normalized direction. For each view (totally 256 views in each scene), say you predict `N` suctions. The result you save for that view should be a `Nx7` numpy array. 

<img src="https://github.com/graspnet/suctionnetAPI/blob/master/suction_definition.jpg" />

## Installation

Please install [Point Cloud Utils](https://github.com/fwilliams/point-cloud-utils) first, then use the following commands

``` 
git clone https://github.com/graspnet/suctionnetAPI
cd suctionnetAPI
pip install .
```

## Evaluation Prerequisite

To evaluate predictions, please make sure you pass the completeness check. Refer `examples/check_and_explore_data.py` to check the completeness.

## Examples

We provide several examples to use our API in folder `examples`

Check, explore and load data:  `examples/check_and_explore_data.py`

Evaluate your results: `examples/evaluation.py`

Visualize data and labels: `visualization.py`

Create dense point clouds: `dense_pcd.py`

## Citation

If you find our work useful,  please cite

```
@ARTICLE{suctionnet,
  author={Cao, Hanwen and Fang, Hao-Shu and Liu, Wenhai and Lu, Cewu},
  journal={IEEE Robotics and Automation Letters}, 
  title={SuctionNet-1Billion: A Large-Scale Benchmark for Suction Grasping}, 
  year={2021},
  volume={6},
  number={4},
  pages={8718-8725},
  doi={10.1109/LRA.2021.3115406}}
```
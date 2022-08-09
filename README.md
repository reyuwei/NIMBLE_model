NIMBLE: A Non-rigid Hand Model with Bones and Muscles
========

![](https://liyuwei.cc/proj/img/nimble_teaser.jpg)

We present NIMBLE, a non-rigid parametric hand model that includes bones and muscles, bringing 3D hand model to a new level of realism. By enforcing the inner bones and muscles to match anatomic and kinematic rules, NIMBLE can animate 3D hands to new poses at unprecedented realism.

To learn about PIANO, please visit our website: https://liyuwei.cc/proj/nimble

You can find the NIMBLE paper at: https://arxiv.org/abs/2202.04533


For comments or questions, please email us at: Yuwei Li (liyw@shanghaitech.edu.cn)

---
## Getting Started:

> python demo.py

## Requirements
1. Pytorch3d
2. Pytorch
3. trimesh (with pyembree for faster collision detection)
4. opencv-python

## Files
1. Blender rendering file
2. Model files

## Acknowledgements:

This model and code was developped and used for the paper *NIMBLE: A Non-rigid Hand Model with Bones and Muscles* for SIGGRAPH22.
See [project page](https://liyuwei.cc/proj/nimble)

It reuses part of the great code from 
[manopth](https://github.com/hassony2/manopth/blob/master/manopth) by [Yana Hasson](https://hassony2.github.io/),
[pytorch_HMR](https://github.com/MandyMo/pytorch_HMR) by [Zhang Xiong](https://github.com/MandyMo) and
[SMPLX](https://github.com/vchoutas/smplx) by [Vassilis Choutas](https://github.com/vchoutas)!


If you find this code useful for your research, consider citing:

```
@article{10.1145/3528223.3530079,
        author = {Li, Yuwei and Zhang, Longwen and Qiu, Zesong and Jiang, 
            Yingwenqi and Li, Nianyi and Ma, Yuexin and Zhang, Yuyao and
            Xu, Lan and Yu, Jingyi},
        title = {NIMBLE: A Non-Rigid Hand Model with Bones and Muscles},
        year = {2022},
        issue_date = {July 2022},
        publisher = {Association for Computing Machinery},
        address = {New York, NY, USA},
        volume = {41},
        number = {4},
        issn = {0730-0301},
        url = {https://doi.org/10.1145/3528223.3530079},
        doi = {10.1145/3528223.3530079},
        journal = {ACM Trans. Graph.},
        month = {jul},
        articleno = {120},
        numpages = {16}
        }
```
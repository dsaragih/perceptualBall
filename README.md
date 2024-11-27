# Towards Semantic Adversarial Perturbations

This is the code for the report: 
[Towards Semantic Adversarial Perturbations](report.pdf) by Daniel Saragih. This was done as part of the course MAT1510 at the University of Toronto. A large part of the code is based on the paper: [Explaining Classifiers using Adversarial Perturbations on the Perceptual Ball](https://arxiv.org/abs/1912.09405) by Andrew Elliott, Stephen Law and Chris Russell.

The primary mechanism can be found in `common_code/` which can be integrated into other pipelines, as well as some additional routines which are useful for plotting etc. We call the functions therein from the `run.py` file, which is the main entry point for the code. To expedite this process, we have made shell scripts `multi_launch_X.sh` which can be used to iterate through experiments.


## Citation
If you use this code for your project please consider citing the original:
```
@inproceedings{Elliott2021PerceptualBall,
  title={Explaining Classifiers using Adversarial Perturbations on the Perceptual Ball},
  author={Elliott, Andrew and Law, Stephen and Russell, Chris},
  booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2021}
}
```


## Other references

In addition to acknowledge the other resources in this field, several of which we use in the paper. 

**Visual Explanation Experiments**
* Insertion Deletion [Rise repo](https://github.com/eclique/RISE) 
* Pointing Game [TorchRay](github.com/facebookresearch/TorchRay)

**Alternative Saliency Methods**
For reference, here are links to alternative saliency methods.
We note that if available we use the standard implementation of each methods in each game and use a reference implementation otherwise. see paper for full details. 

* [TorchRay Library - Multiple methods](github.com/facebookresearch/TorchRay)
* [Pytorch Cnn Visualisation - Multiple methods](https://github.com/utkuozbulak/pytorch-cnn-visualizations)
* [NormGrad](www.github.com/ruthcfong/TorchRay/tree/normgrad)
* [RISE](github.com/eclique/RISE)

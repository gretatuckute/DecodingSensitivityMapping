# SVM-decoding-and-sensitivity-mapping

Toolbox for computing and visualizing sensitivity maps of EEG-based Support Vector Machines (SVM) based on an approach originally proposed by [Rasmussen et al., 2011](https://www.sciencedirect.com/science/article/pii/S1053811910016198) was adapted. More information found in [Single Trial Decoding of Scalp EEG Under Naturalistic Stimuli](https://www.biorxiv.org/content/early/2018/11/29/481630).

<a href="https://www.codecogs.com/eqnedit.php?latex=k(\mathbf{x}_n,\mathbf{x})&space;=&space;\exp{(-\gamma||\mathbf{x}_n-\mathbf{x}||^2)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?k(\mathbf{x}_n,\mathbf{x})&space;=&space;\exp{(-\gamma||\mathbf{x}_n-\mathbf{x}||^2)}" title="k(\mathbf{x}_n,\mathbf{x}) = \exp{(-\gamma||\mathbf{x}_n-\mathbf{x}||^2)}" /></a>

The sensitivity map is computed as the derivative of the Radial Basis Function (RBF) kernel:

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;\alpha^{\top}\mathbf{k_x}}{\partial&space;x_j}&space;=&space;\sum_n&space;\alpha_n&space;2&space;\gamma&space;(x_{n,j}-x_j)&space;\exp(-\gamma\left\lVert\mathbf{x_\mathbf{n}}-\mathbf{x}\right\rVert^2)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;\alpha^{\top}\mathbf{k_x}}{\partial&space;x_j}&space;=&space;\sum_n&space;\alpha_n&space;2&space;\gamma&space;(x_{n,j}-x_j)&space;\exp(-\gamma\left\lVert\mathbf{x_\mathbf{n}}-\mathbf{x}\right\rVert^2)" title="\frac{\partial \alpha^{\top}\mathbf{k_x}}{\partial x_j} = \sum_n \alpha_n 2 \gamma (x_{n,j}-x_j) \exp(-\gamma\left\lVert\mathbf{x_\mathbf{n}}-\mathbf{x}\right\rVert^2)" /></a>


The Python toolbox [Scikit-learn](https://scikit-learn.org/stable/) was used to implement SVM models.

If you find this implementation useful, please cite: [Single Trial Decoding of Scalp EEG Under Naturalistic Stimuli](https://www.biorxiv.org/content/early/2018/11/29/481630)

@article{Tuckute2018,

  title={Single Trial Decoding of Scalp EEG Under Naturalistic Stimuli},
  
  author={Tuckute, Greta and Hansen, Sofie Therese and Pedersen, Nicolai and Steenstrup, Dea and Hansen, Lars Kai},
  
  journal={arXiv preprint arXiv:10.1101/481630},
  
  year={2018}
}

*In collaboration with Lars Kai Hansen and Sofie T. Hansen, Technical University of Denmark, 2018*









Code for analyzing EEG, implementation and optimization of Support Vector Machines (SVM), computation of EEG-based sensitivity mapping


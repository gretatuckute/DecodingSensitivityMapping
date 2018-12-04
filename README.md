### SVM decoding and sensitivity mapping

Toolbox for computing and visualizing sensitivity maps of EEG-based Support Vector Machines (SVM) based on an approach originally proposed by [Rasmussen et al., 2011](https://www.sciencedirect.com/science/article/pii/S1053811910016198). More information found in [Single Trial Decoding of Scalp EEG Under Naturalistic Stimuli](https://www.biorxiv.org/content/early/2018/11/29/481630). The SVM classifier uses the Radial Basis Function (RBF) kernel:

<a href="https://www.codecogs.com/eqnedit.php?latex=k(\mathbf{x}_n,\mathbf{x})&space;=&space;\exp{(-\gamma||\mathbf{x}_n-\mathbf{x}||^2)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?k(\mathbf{x}_n,\mathbf{x})&space;=&space;\exp{(-\gamma||\mathbf{x}_n-\mathbf{x}||^2)}" title="k(\mathbf{x}_n,\mathbf{x}) = \exp{(-\gamma||\mathbf{x}_n-\mathbf{x}||^2)}" /></a>

The sensitivity map is computed as the derivative of the RBF kernel:

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;\alpha^{\top}\mathbf{k_x}}{\partial&space;x_j}&space;=&space;\sum_n&space;\alpha_n&space;2&space;\gamma&space;(x_{n,j}-x_j)&space;\exp(-\gamma\left\lVert\mathbf{x_\mathbf{n}}-\mathbf{x}\right\rVert^2)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;\alpha^{\top}\mathbf{k_x}}{\partial&space;x_j}&space;=&space;\sum_n&space;\alpha_n&space;2&space;\gamma&space;(x_{n,j}-x_j)&space;\exp(-\gamma\left\lVert\mathbf{x_\mathbf{n}}-\mathbf{x}\right\rVert^2)" title="\frac{\partial \alpha^{\top}\mathbf{k_x}}{\partial x_j} = \sum_n \alpha_n 2 \gamma (x_{n,j}-x_j) \exp(-\gamma\left\lVert\mathbf{x_\mathbf{n}}-\mathbf{x}\right\rVert^2)" /></a>

The Python toolbox [Scikit-learn](https://scikit-learn.org/stable/) is used to implement SVM models.

### Script information 

The script, *computeSensitivityMap.py* fits a SVM classifier based on the input data matrix and label array, and computes the corresponding sensitivity map. 

#### Inputs 
- X: EEG data 2d matrix containing trials as rows, and features (channels * time points) as columns.
- y: List/NumPy array containing binary class labels, y = {-1, 1}.
- C: SVM classifier regularization parameter. 
- Gamma: Free parameter of the RBF kernel, SVM classifier.

#### Outputs
- s_matrix: sensitivity map matrix.
- plt: Visualization of the sensitivity map.

#### Example function call
computeSensitivityMap(X, y, C_val = 1, gamma_val = 0.0005, no_channels = 32, no_timepoints = 60)

#### Example output 
![alt text](https://raw.githubusercontent.com/gretatuckute/DecodingSensitivityMapping/master/Example/sensitivity_map.png)

Example of a sensitivity map computed based on a SVM classifier separating animate/inanimate visual stimuli. The EEG signal was bandpass filtered to 1-25 Hz and downsampled to 100 Hz. Epochs of 600 ms (100 ms pre and 500 ms post stimulus onset) were extracted.

### Other scripts in this repository 
- runSVMcrossvalidation.py: Script for SVM cross-validation (hyperparameters: C and gamma) in a leave-one-subject-out approach. Runs from the command-line with the subject number as the cmd flag. Data has to be loaded in the Python script.
- NPAIRS_main.py: Implementation of the NPAIRS [(Strother et al., 2011)](https://www.sciencedirect.com/science/article/pii/S1053811901910341?via%3Dihub) cross-validation framework for estimation of sensitivity map visualization uncertainty. 
- NPAIRS_functions.py: Functions called in NPAIRS_main.py.

### Reference
If you find this implementation useful, please cite: [Single Trial Decoding of Scalp EEG Under Naturalistic Stimuli](https://www.biorxiv.org/content/early/2018/11/29/481630)

@article{Tuckute2018,
  title={Single Trial Decoding of Scalp EEG Under Naturalistic Stimuli},
  author={Tuckute, Greta and Hansen, Sofie Therese and Pedersen, Nicolai and Steenstrup, Dea and Hansen, Lars Kai},
  journal={arXiv preprint arXiv:10.1101/481630},
  year={2018}
}

*In collaboration with Lars Kai Hansen and Sofie T. Hansen, Technical University of Denmark, 2018*









Code for analyzing EEG, implementation and optimization of Support Vector Machines (SVM), computation of EEG-based sensitivity mapping


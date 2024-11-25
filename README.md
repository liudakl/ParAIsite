# ParAIsite: a Fine-Tuned Neural Network model for Predicting Thermal Conductivity


<div align="center">
<img src="https://github.com/liudakl/fine_tuning_papers/blob/main/paper/logo.png?raw=true" alt="Model" width="400" height="400">
</div>


In this study, we introduce **ParAIsite**, a deep learning model designed for predicting thermal conductivity of materials. Machine learning promises to accelerate the material discovery by enabling high-throughput prediction of desirable macro-properties from atomic-level descriptors or structures. However, the limited data available about precise values of these properties have been a barrier, leading to predictive models with limited precision or ability to generalize. This is particularly true of lattice thermal conductivity (LTC): existing datasets of precise (ab initio, DFT-based) computed values are limited to a few dozen materials with little variability. Based on such datasets, we study the impact of transfer learning on both the precision and generalizability of a deep learning model (ParAIsite). We start from an existing model (MEGNet[1]) and show that improvements are obtained by fine-tuning a pretrained version of it on a different tasks. Interestingly, we also show that a much greater improvement is obtained when first fine-tuning it on a large datasets of low-quality approximations of LTC (based on the AGL model), and then applying a second phase of fine-tuning with our high-quality, smaller-scale datasets. The promising results obtained pave the way not only towards a greater ability to explore large databases in search of low thermal conductivity materials but also to methods enabling increasingly precise predictions in areas where quality data are rare. 


## Methodology

Steps that we followed to achieve the results:

1. **Preprocessing Data**
   - Clean and format datasets (ie. we need structure and compounds to be ready before the execution)
   - Merge datasets from different sources
2. **Model Development**
   - Fine-tune pre-trained MEGNET model
   - Develop and test new architectures of our MLP model
3. **Evaluation**
   - Assess model performance
   - Compare with baseline models

## Results

- Improved accuracy in predicting thermal conductivity
- Demonstrated potential for application in materials science

## References

```txt
1. Chen, C.; Ye, W.; Zuo, Y.; Zheng, C.; Ong, S. P. Graph Networks as a Universal Machine Learning Framework for
Molecules and Crystals. Chem. Mater. 2019, 31 (9), 3564–3572. https://doi.org/10.1021/acs.chemmater.9b01294.
```

## How to run ParAIsite

Clone the project to your machine:

```bash
  git clone https://github.com/liudakl/ParAIsite.git 
```

Go to the project directory

```bash
  cd ParAIsite/main/src
```

If you would like to reproduce training, please do:

```bash
  python 3.10 ParAIsite_train.py
```
Please keep in mind that you need to select in the script on which dataset you would like perform training; the best architecture of MLP model, and etc. To be able reproduce the results, please keep the selections as they are.

If you would like to reproduce training on AFLOW and after additional train on another datasets, please do:

```bash
  python 3.10 ParAIsite_double_train.py
```
Please keep in mind that you need to select in the script on which dataset you would like perform training; the best architecture of MLP model, and etc. To be able reproduce the results, please keep the selections as they are.


### Which data ParAIsite expects you to have before the execution it?

- Requires targets (Thermal Conductivities) and inputs (structure of compounds) in pkl format; 
- For the datasets: AFLOW, Dataset1, Dataset2, and MIX they are avaible in *structure_scaler* folder.

## How to test ParAIsite on Data?

Testings ParAIsite on Data are already integrated inside the code for model training/double training. 
Please keep in mind that you need to change the script with respect on which dataset you would like perform test.


## Summary: Validation Results (Dataset on Train vs Dataset on Validation)


| Train on \ Test on | Dataset1              | Dataset2            | MIX              | AFLOW           |
|--------------------|------------------|------------------|------------------|-----------------|
| **Step I: No weights MEGNET** |                |                  |                  |                 |
| Dataset1                | 0.82 (0.21)      | 2.53 (2.17)      | 1.74 (1.26)      | 2.16 (1.26)     |
| Dataset2              | 0.51 (0.09)      | 0.42 (0.07)      | 0.43 (0.05)      | 0.48 (0.05)     |
| MIX                | 0.69 (0.15)      | 0.77 (0.15)      | 0.83 (0.07)      | 0.97 (0.27)     |
| AFLOW              | 0.55 (0.34)      | 1.18 (0.27)      | 0.93 (0.27)      | 0.62 (0.33)     |
| **Step II: With weights MEGNET** |            |                  |                  |                 |
| Dataset1                | 0.76 (0.29)      | 3.09 (2.40)      | 2.07 (1.40)      | 2.32 (1.94)     |
| Dataset2              | 0.55 (0.14)      | 0.42 (0.06)      | 0.44 (0.04)      | 0.52 (0.10)     |
| MIX                | 0.66 (0.14)      | 0.75 (0.16)      | 0.81 (0.11)      | 1.10 (0.47)     |
| AFLOW              | 0.44 (0.14)      | 1.27 (0.48)      | 0.94 (0.30)      | 0.50 (0.03)     |
| **Step III: Retrained on AFLOW** |    |                  |                  |                 |
| Dataset1                | 0.34 (0.15)      | 1.26 (0.44)      | 0.87 (0.27)      | 0.57 (0.10)     |
| Dataset2              | 0.55 (0.10)      | 0.78 (0.20)      | 0.63 (0.18)      | 0.62 (0.07)     |
| MIX                | 0.37 (0.14)      | 0.78 (0.31)      | 0.69 (0.19)      | 0.59 (0.06)     |




# (step I) ParAIsite WITHOUT MEGNET weitghts:

## Training Results
| Dataset | Validation Error (Mean ± Std) |
|---------|-------------------------------|
| Dataset1     |  0.54 (0.22)                  |
| Dataset2   |  0.38 (0.05)                 |
| MIX     |  0.67 (0.10)                  |
| AFLOW   |  0.60 (0.31)             	  |

## Validation Results

### Test on Dataset1
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| Dataset1     |  2.53 (2.17)                      |
| Dataset2       |  1.74 (1.26)             		|
| AFLOW     |  2.16 (1.26)            		|

### Test  on Dataset2
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| Dataset1       |  0.51 (0.09)                     	|
| MIX       |  0.43 (0.05)            		|
| AFLOW     |  0.48 (0.05)        		|

### Test  on MIX
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| Dataset1       | 0.69 (0.15)             		|
| Dataset2     | 0.77 (0.15)              		|
| AFLOW     | 0.97 (0.27)      			|

### Test  on AFLOW

| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| Dataset1       |  0.55 (0.34)                    	|
| Dataset2     |  1.18 (0.27)                    	|
| MIX       |  0.93 (0.27)        		|



# (step II) ParAIsite with MEGNET weitghts:

## Training Results
| Dataset | Validation Error (Mean ± Std) |
|---------|-------------------------------|
| Dataset1     |  0.51 (0.24)                  |
| Dataset2   |  0.37 (0.08)                  |
| MIX     |  0.71 (0.12)                  |
| AFLOW   |  0.49 (0.03)                  |

## Validation Results

### Test on Dataset1
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| Dataset2     | 3.09 (2.40)             		|
| MIX       | 2.07 (1.40)              		|
| AFLOW     | 2.32 (1.94)    			|

### Test  on Dataset2
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| Dataset1       | 0.55 (0.14)              		|
| MIX       | 0.44 (0.04)              		|
| AFLOW     | 0.52 (0.10)      			|

### Test  on MIX
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| Dataset1       | 0.66 (0.14)              		|
| Dataset2     | 0.75 (0.16)                       |
| AFLOW     | 1.10 (0.47)     			|

### Test  on AFLOW

| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| Dataset1       |  0.44 (0.14)             		|
| Dataset2     |  1.27 (0.48)             		|
| MIX       |  0.94 (0.30)            		|




# (step III) ParAIsite: 1) trained on AFLOW with MEGNET weights (the best model from step II ) + 2) re-trained on the datasets:

## Training Results
| Dataset | Validation Error (Mean ± Std) |
|---------|-------------------------------|
| Dataset1     |  0.29 (0.13)                  |
| Dataset2   |  0.69 (0.26)                  |
| MIX     |  0.62 (0.20)                  |

## Validation Results

### Test  on Dataset1
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| Dataset2     |  1.26 (0.44)             		|
| MIX       |  0.87 (0.27)            		|
| AFLOW     |  0.57 (0.10)     			|

### Test  on Dataset2
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| Dataset1       |  0.55 (0.10)                     	|
| MIX       |  0.63 (0.18)                      |
| AFLOW     |  0.62 (0.07)            		|

### Test  on MIX
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| Dataset1       |  0.37 (0.14)            		|
| Dataset2     |  0.78 (0.31)          		|
| AFLOW     |  0.59 (0.06)    			|


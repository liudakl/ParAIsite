# ParAIsite: a Fine-Tuned Neural Network model for Predicting Thermal Conductivity


<div align="center">
<img src="https://github.com/liudakl/fine_tuning_papers/blob/main/paper/paraisite_2.png?raw=true" alt="Model" width="400" height="400">
</div>

<div style="text-align: justify;">
In this study, we introduce ParAIsite, a deep learning model designed for predicting thermal conductivity of materials. Despite the existence of large databases, accessing extensive thermal property data, such as thermal conductivity for various compounds, remains challenging. One effective approach for achieving precise predictions on small datasets is using the framework of Transfer Learning (TL). In TL, a pre-trained machine learning (ML) model on a large dataset can be fine-tuned (FT) on a smaller dataset. In our case, we utilized an existing MEGNET [1] model pre-trained on 62315 Materials Project formation energy as of 2018.6.1 data, extracted the output layer, and inserted our multi-layer perceptron (MLP) model on top of it, making its output the input for our model. This innovative approach demonstrates the potential of ParAIsite to significantly advance computational materials science by providing accurate thermal conductivity predictions.
</div>

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
  git clone https://github.com/liudakl/fine_tuning_papers
```

Go to the project directory

```bash
  cd https://github.com/liudakl/fine_tuning_papers/tree/main/paper/megnet_p31/pytorch/matgl-main/src
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

- Requires targets (Thermal Conductivities) and inputs (structure of compounds) in pkl format; For the datasets: AFLOW, L96, MIX and HH143 they are avaible in *structure_scaler* folder.

## How to test ParAIsite on Data?

If you would like to test your model on datasets, please do:

```bash
  python 3.10 test_model.py
```
Please keep in mind that you need to select in the script on which model and on which dataset you would like perform test.


# ParAIsite  Training and Testing Results:

## Training Results
| Dataset | Validation Error (Mean ± Std) |
|---------|-------------------------------|
| L96     | 0.47 (0.20)                   |
| HH143   | 0.34 (0.07)                   |
| MIX     | 0.50 (0.11)                   |

## Testing Results

### Test on L96
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| HH143     | 1.45 (0.47)                       |
| MIX       | 1.01 (0.24)                       |
| L96 (validation set) | 0.47 (0.20)            |
| L96 (full set)       | 0.34 (0.19)            |

### Test  on HH143
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| L96       | 0.97 (0.08)                       |
| MIX       | 0.52 (0.06)                       |
| HH143 (validation set) | 0.34 (0.07)          |
| HH143 (full set)       | 0.22 (0.10)          |

### Test  on MIX
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| L96       | 0.47 (0.21)                       |
| HH143     | 0.27 (0.11)                       |
| MIX (validation set) | 0.50 (0.11)            |
| MIX (full set)       | 0.35 (0.15)            |


# ParAIsite trained on AFLOW Training and Testing Results:

## Training Results
| Dataset | Validation Error (Mean ± Std) |
|---------|-------------------------------|
| AFLOW     |   0.49 (0.02)                |


## Testing Results
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| L96       |    0.82 (0.02)                   |
| HH143     |       0.55 (0.04)                 |
| MIX     |   0.66 (0.03)                |
| AFLOW (full set)       |   0.34 (0.05)           |



# ParAIsite  trained on AFLOW + datasets Training and Testing Results:

## Training Results
| Dataset | Validation Error (Mean ± Std) |
|---------|-------------------------------|
| L96     |   0.29 (0.10)                |
| HH143   |   0.61 (0.15)                 |
| MIX     |   0.42 (0.06)                 |

## Testing Results

### Test  on L96
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| HH143     | 3.02 (0.58)                  |
| MIX       |   1.91 (0.34)                     |
| L96 (full set)  |      0.23 (0.08)               |

### Test  on HH143
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| L96       |      0.78 (0.04)                 |
| MIX       |         0.45 (0.06)              |
| HH143 (full set)    |   0.23 (0.10)          |

### Test  on MIX
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| L96       |          0.33 (0.17)             |
| HH143     |           0.24 (0.10)             |
| MIX (full set)  |     0.27 (0.13)             |

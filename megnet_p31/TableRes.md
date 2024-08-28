# ParAIsite: a Fine-Tuned Neural Network model for Predicting Thermal Conductivity

## Summary Table: Dataset on Train vs Dataset on Validation

| Train on \ Test on   | L96 (val) | L96 (full) | HH143 (val) | HH143 (full) | MIX (val) | MIX (full) | AFLOW (val) | AFLOW (full) |
|----------------------|-----------|------------|-------------|--------------|-----------|------------|-------------|--------------|
| **Step I: No MEGNET**|           |            |             |              |           |            |             |              |
| L96                  | 0.51 (0.20)| 0.33 (0.23) | 1.50 (0.79)  | -            | 1.03 (0.45)| -          | -           | -            |
| HH143                | 0.95 (0.09)| -          | 0.35 (0.07)  | 0.28 (0.08)   | 0.55 (0.04)| -          | -           | -            |
| MIX                  | 0.50 (0.24)| -          | 0.25 (0.11)  | -            | 0.48 (0.11)| 0.35 (0.16)| -           | -            |
| AFLOW                | 0.90 (0.11)| -          | 0.69 (0.19)  | -            | 0.77 (0.16)| -          | 0.71 (0.34) | 0.64 (0.42)  |
| **Step II: With MEGNET** |      |            |             |              |           |            |             |              |
| L96                  | 0.47 (0.20)| 0.34 (0.19) | 1.45 (0.47)  | -            | 1.01 (0.24)| -          | -           | -            |
| HH143                | 0.97 (0.08)| -          | 0.34 (0.07)  | 0.22 (0.10)   | 0.52 (0.06)| -          | -           | -            |
| MIX                  | 0.47 (0.21)| -          | 0.27 (0.11)  | -            | 0.50 (0.11)| 0.35 (0.15)| -           | -            |
| AFLOW                | 0.82 (0.02)| -          | 0.55 (0.04)  | -            | 0.66 (0.03)| -          | -           | 0.34 (0.05)  |
| **Step III: Best Model from II, Re-trained** | | |             |              |           |            |             |              |
| L96                  | -         | 0.23 (0.08) | 3.02 (0.58)  | -            | 1.91 (0.34)| -          | -           | -            |
| HH143                | 0.78 (0.04)| -          | -            | 0.23 (0.10)   | 0.45 (0.06)| -          | -           | -            |
| MIX                  | 0.33 (0.17)| -          | 0.24 (0.10)  | -            | -         | 0.27 (0.13)| -           | -            |


# (step I) ParAIsite WITHOUT MEGNET weitghts:

## Training Results
| Dataset | Validation Error (Mean ± Std) |
|---------|-------------------------------|
| L96     | 0.51 (0.20)                   |
| HH143   | 0.35 (0.07)                  |
| MIX     | 0.48 (0.11)                   |
| AFLOW     |  0.71 (0.34)              |

## Validation Results

### Test on L96
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| HH143     |    1.50 (0.79)                    |
| MIX       |    1.03 (0.45)                    |
| L96       |  0.33 (0.23)            		|
| AFLOW     |    7.73 (3.47)           			|

### Test  on HH143
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| L96       | 0.95 (0.09)                      |
| MIX       | 0.55 (0.04)                        |
| HH143        |   0.28 (0.08)         |
| AFLOW       |              	|

### Test  on MIX
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| L96       | 0.50 (0.24)                       |
| HH143     |  0.25 (0.11)                      |
| MIX       |   0.35 (0.16)          |
| AFLOW       |              |

### Test  on AFLOW

| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| L96       | 0.90 (0.11)                     |
| HH143     |  0.69 (0.19)                      |
| MIX     |    0.77 (0.16)              |
| AFLOW        |     0.64 (0.42)         |


# (step II) ParAIsite with MEGNET weitghts:

## Training Results
| Dataset | Validation Error (Mean ± Std) |
|---------|-------------------------------|
| L96     | 0.47 (0.20)                   |
| HH143   | 0.34 (0.07)                   |
| MIX     | 0.50 (0.11)                   |
| AFLOW     |   0.49 (0.02)               |

## Validation Results

### Test on L96
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| HH143     | 1.45 (0.47)                       |
| MIX       | 1.01 (0.24)                       |
| L96       | 0.34 (0.19)            |

### Test  on HH143
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| L96       | 0.97 (0.08)                       |
| MIX       | 0.52 (0.06)                       |
| HH143        | 0.22 (0.10)          |

### Test  on MIX
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| L96       | 0.47 (0.21)                       |
| HH143     | 0.27 (0.11)                       |
| MIX       | 0.35 (0.15)            |

### Test  on AFLOW

| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| L96       |    0.82 (0.02)                   |
| HH143     |       0.55 (0.04)                 |
| MIX     |   0.66 (0.03)                |
| AFLOW       |   0.34 (0.05)           |



# (step III) ParAIsite: 1) trained on AFLOW with MEGNET weights (the best model from step II ) + 2) re-trained on the datasets:

## Training Results
| Dataset | Validation Error (Mean ± Std) |
|---------|-------------------------------|
| L96     |   0.29 (0.10)                |
| HH143   |   0.61 (0.15)                 |
| MIX     |   0.42 (0.06)                 |

## Validation Results

### Test  on L96
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| HH143     | 3.02 (0.58)                  |
| MIX       |   1.91 (0.34)                     |
| L96  |      0.23 (0.08)               |

### Test  on HH143
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| L96       |      0.78 (0.04)                 |
| MIX       |         0.45 (0.06)              |
| HH143   |   0.23 (0.10)          |

### Test  on MIX
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| L96       |          0.33 (0.17)             |
| HH143     |           0.24 (0.10)             |
| MIX  |     0.27 (0.13)             |

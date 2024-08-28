# ParAIsite: a Fine-Tuned Neural Network model for Predicting Thermal Conductivity

# (step I) ParAIsite WITHOUT MEGNET weitghts:

## Training Results
| Dataset | Validation Error (Mean ± Std) |
|---------|-------------------------------|
| L96     | 0.51 (0.20)                   |
| HH143   | 0.35 (0.07)                  |
| MIX     | 0.48 (0.11)                   |
| AFLOW     |  0.71 (0.34)              |

## Testing Results

### Test on L96
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| HH143     |                        |
| MIX       |                        |
| L96 (validation set) |            |
| L96 (full set)       |             |

### Test  on HH143
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| L96       |                     |
| MIX       |                        |
| HH143 (validation set) |          |
| HH143 (full set)       |           |

### Test  on MIX
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| L96       |                        |
| HH143     |                        |
| MIX (validation set) |             |
| MIX (full set)       |             |

### Test  on AFLOW

| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| L96       |                      |
| HH143     |                        |
| MIX     |                 |
| AFLOW (validation set)       |   )           |
| AFLOW (full set)       |   )           |


# (step II) ParAIsite with MEGNET weitghts:

## Training Results
| Dataset | Validation Error (Mean ± Std) |
|---------|-------------------------------|
| L96     | 0.47 (0.20)                   |
| HH143   | 0.34 (0.07)                   |
| MIX     | 0.50 (0.11)                   |
| AFLOW     |   0.49 (0.02)               |

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

### Test  on AFLOW

| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| L96       |    0.82 (0.02)                   |
| HH143     |       0.55 (0.04)                 |
| MIX     |   0.66 (0.03)                |
| AFLOW (full set)       |   0.34 (0.05)           |



# (step III) ParAIsite: 1) trained on AFLOW with MEGNET weights (the best model from step II ) + 2) re-trained on the datasets:

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

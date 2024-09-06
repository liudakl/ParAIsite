# ParAIsite: a Fine-Tuned Neural Network model for Predicting Thermal Conductivity

## Summary: Validation Results (Dataset on Train vs Dataset on Validation)


| Train on \ Test on | L96              | HH143            | MIX              | AFLOW           |
|--------------------|------------------|------------------|------------------|-----------------|
| **Step I: No weights MEGNET** |                |                  |                  |                 |
| L96                | 0.54 (0.22)      | 2.53 (2.17)      | 1.74 (1.26)      | 2.16 (1.26)     |
| HH143              | 0.51 (0.09)      | 0.38 (0.05)      | 0.43 (0.05)      | 0.48 (0.05)     |
| MIX                | 0.69 (0.15)      | 0.77 (0.15)      | 0.67 (0.10)      | 0.97 (0.27)     |
| AFLOW              | 0.55 (0.34)      | 1.18 (0.27)      | 0.93 (0.27)      | 0.60 (0.31)     |
| **Step II: With weights MEGNET** |            |                  |                  |                 |
| L96                | 0.51 (0.24)      | 3.09 (2.40)      | 2.07 (1.40)      | 2.32 (1.94)     |
| HH143              | 0.55 (0.14)      | 0.37 (0.08)      | 0.44 (0.04)      | 0.52 (0.10)     |
| MIX                | 0.66 (0.14)      | 0.75 (0.16)      | 0.71 (0.12)      | 1.10 (0.47)     |
| AFLOW              | 0.44 (0.14)      | 1.27 (0.48)      | 0.94 (0.30)      | 0.49 (0.03)     |
| **Step III: Retrained on AFLOW** |    |                  |                  |                 |
| L96                | 0.29 (0.13)      | 1.26 (0.44)      | 0.87 (0.27)      | 0.57 (0.10)     |
| HH143              | 0.55 (0.10)      | 0.69 (0.26)      | 0.63 (0.18)      | 0.62 (0.07)     |
| MIX                | 0.37 (0.14)      | 0.78 (0.31)      | 0.62 (0.20)      | 0.59 (0.06)     |




# (step I) ParAIsite WITHOUT MEGNET weitghts:

## Training Results
| Dataset | Validation Error (Mean ± Std) |
|---------|-------------------------------|
| L96     |  0.54 (0.22)                  |
| HH143   |  0.38 (0.05)                 |
| MIX     |  0.67 (0.10)                  |
| AFLOW   |  0.60 (0.31)             	  |

## Validation Results

### Test on L96
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| HH143     |  2.53 (2.17)                      |
| MIX       |  1.74 (1.26)             		|
| AFLOW     |  2.16 (1.26)            		|

### Test  on HH143
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| L96       |  0.51 (0.09)                     	|
| MIX       |  0.43 (0.05)            		|
| AFLOW     |  0.48 (0.05)        		|

### Test  on MIX
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| L96       | 0.69 (0.15)             		|
| HH143     | 0.77 (0.15)              		|
| AFLOW     | 0.97 (0.27)      			|

### Test  on AFLOW

| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| L96       |  0.55 (0.34)                    	|
| HH143     |  1.18 (0.27)                    	|
| MIX       |  0.93 (0.27)        		|



# (step II) ParAIsite with MEGNET weitghts:

## Training Results
| Dataset | Validation Error (Mean ± Std) |
|---------|-------------------------------|
| L96     |  0.51 (0.24)                  |
| HH143   |  0.37 (0.08)                  |
| MIX     |  0.71 (0.12)                  |
| AFLOW   |  0.49 (0.03)                  |

## Validation Results

### Test on L96
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| HH143     | 3.09 (2.40)             		|
| MIX       | 2.07 (1.40)              		|
| AFLOW     | 2.32 (1.94)    			|

### Test  on HH143
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| L96       | 0.55 (0.14)              		|
| MIX       | 0.44 (0.04)              		|
| AFLOW     | 0.52 (0.10)      			|

### Test  on MIX
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| L96       | 0.66 (0.14)              		|
| HH143     | 0.75 (0.16)                       |
| AFLOW     | 1.10 (0.47)     			|

### Test  on AFLOW

| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| L96       |  0.44 (0.14)             		|
| HH143     |  1.27 (0.48)             		|
| MIX       |  0.94 (0.30)            		|




# (step III) ParAIsite: 1) trained on AFLOW with MEGNET weights (the best model from step II ) + 2) re-trained on the datasets:

## Training Results
| Dataset | Validation Error (Mean ± Std) |
|---------|-------------------------------|
| L96     |  0.29 (0.13)                  |
| HH143   |  0.69 (0.26)                  |
| MIX     |  0.62 (0.20)                  |

## Validation Results

### Test  on L96
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| HH143     |  1.26 (0.44)             		|
| MIX       |  0.87 (0.27)            		|
| AFLOW     |  0.57 (0.10)     			|

### Test  on HH143
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| L96       |  0.55 (0.10)                     	|
| MIX       |  0.63 (0.18)                      |
| AFLOW     |  0.62 (0.07)            		|

### Test  on MIX
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| L96       |  0.37 (0.14)            		|
| HH143     |  0.78 (0.31)          		|
| AFLOW     |  0.59 (0.06)    			|

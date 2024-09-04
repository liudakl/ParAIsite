# ParAIsite: a Fine-Tuned Neural Network model for Predicting Thermal Conductivity

## Summary: Validation Results (Dataset on Train vs Dataset on Validation)

| Train on \ Test on   | L96            | HH143          | MIX            | AFLOW          |
|----------------------|----------------|----------------|----------------|----------------|
| **Step I: No WEIGHTS MEGNET**|                |                |                |                |
| L96                  | 0.46 (0.17)    | 2.24 (0.96)    | 1.53 (0.55)    | 2.45 (0.67)    |
| HH143                | 0.53 (0.10)    | 0.37 (0.07)    | 0.44 (0.04)    | 0.50 (0.10)    |
| MIX                  | 0.70 (0.20)    | 0.73 (0.10)    | 0.72 (0.14)    | 1.03 (0.41)    |
| AFLOW                | 0.49 (0.08)    | 1.28 (0.51)    | 0.96 (0.30)    | 0.51 (0.10)    |

| **Step II: With WEIGHTS MEGNET** |            |                |                |                |
| L96                  | 0.52 (0.23)    | 1.83 (1.40)    | 1.30 (0.81)    | 2.44 (1.25)    |
| HH143                | 0.51 (0.11)    | 0.37 (0.08)    | 0.43 (0.06)    | 0.50 (0.07)    |
| MIX                  | 0.63 (0.20)    | 0.76 (0.08)    | 0.71 (0.10)    | 1.16 (0.46)    |
| AFLOW                | 0.42 (0.17)    | 1.17 (0.45)    | 0.88 (0.27)    | 0.51 (0.06)    |

| **Step III: Best Model AFLOW from II, Re-trained** | |          |                |                |
| L96                  | 0.29 (0.08)    | 1.21 (0.33)    | 0.85 (0.20)    | 0.61 (0.20)    |
| HH143                | 0.58 (0.12)    | 0.65 (0.22)    | 0.63 (0.15)    | 0.63 (0.07)    |
| MIX                  | 0.34 (0.10)    | 0.71 (0.23)    | 0.56 (0.16)    | 0.72 (0.28)    |






# (step I) ParAIsite WITHOUT MEGNET weitghts:

## Training Results
| Dataset | Validation Error (Mean ± Std) |
|---------|-------------------------------|
| L96     | 0.46 (0.17)                   |
| HH143   | 0.37 (0.07)                   |
| MIX     | 0.72 (0.14)                   |
| AFLOW   | 0.51 (0.10)              	  |

## Validation Results

### Test on L96
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| HH143     |  2.24 (0.96)                      |
| MIX       |  1.53 (0.55)             		|
| AFLOW     |  2.45 (0.67)            		|

### Test  on HH143
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| L96       | 0.53 (0.10)                      	|
| MIX       | 0.44 (0.04)             		|
| AFLOW     | 0.50 (0.10)         		|

### Test  on MIX
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| L96       |  0.70 (0.20)            		|
| HH143     |  0.73 (0.10)             		|
| AFLOW     |  1.03 (0.41)     			|

### Test  on AFLOW

| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| L96       | 0.49 (0.08)                     	|
| HH143     | 1.28 (0.51)                     	|
| MIX       | 0.96 (0.30)         		|



# (step II) ParAIsite with MEGNET weitghts:

## Training Results
| Dataset | Validation Error (Mean ± Std) |
|---------|-------------------------------|
| L96     | 0.52 (0.23)                   |
| HH143   | 0.37 (0.08)                   |
| MIX     | 0.71 (0.10)                   |
| AFLOW   | 0.51 (0.06)                   |

## Validation Results

### Test on L96
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| HH143     | 1.83 (1.40)             		|
| MIX       | 1.30 (0.81)              		|
| AFLOW     | 2.44 (1.25)    			|

### Test  on HH143
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| L96       | 0.51 (0.11)              		|
| MIX       | 0.43 (0.06)              		|
| AFLOW     | 0.50 (0.07)     			|

### Test  on MIX
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| L96       |   0.63 (0.20)            		|
| HH143     |   0.76 (0.08)                     |
| AFLOW     |   1.16 (0.46)   			|

### Test  on AFLOW

| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| L96       |  0.42 (0.17)             		|
| HH143     |  1.17 (0.45)             		|
| MIX       |  0.88 (0.27)            		|




# (step III) ParAIsite: 1) trained on AFLOW with MEGNET weights (the best model from step II ) + 2) re-trained on the datasets:

## Training Results
| Dataset | Validation Error (Mean ± Std) |
|---------|-------------------------------|
| L96     |  0.29 (0.08)                  |
| HH143   |  0.65 (0.22)                  |
| MIX     |  0.56 (0.16)                  |

## Validation Results

### Test  on L96
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| HH143     |  1.21 (0.33)             		|
| MIX       |  0.85 (0.20)            		|
| AFLOW     |  0.61 (0.20)     			|

### Test  on HH143
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| L96       |   0.58 (0.12)                    	|
| MIX       |   0.63 (0.15)                     |
| AFLOW     |   0.63 (0.07)           		|

### Test  on MIX
| Tested on | Error (Mean ± Std)                |
|-----------|-----------------------------------|
| L96       |   0.34 (0.10)           		|
| HH143     |   0.71 (0.23)          		|
| AFLOW     |   0.72 (0.28)   			|

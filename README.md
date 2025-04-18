# ParAIsite: a Fine-Tuned Neural Network model for Predicting Thermal Conductivity


<div align="center">
<img src="https://github.com/liudakl/fine_tuning_papers/blob/main/paper/logo.png?raw=true">
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

## Results: 

### Validation Results (Dataset on Train vs Dataset on Validation)


## Metric: MAPE (Mean Average Percentage Error) ===> Our Loss Function 

| Train on \ Test on | Dataset1 | Dataset2 | MIX | AFLOW |
|--------------------|------------------|------------------|------------------|------------------|
| **Step I: No weights MEGNET** |  |  |  |  |
| Dataset1           | 0.55 (0.20) | 2.24 (1.15) | 1.57 (0.64) | 2.28 (1.12) |
| Dataset2           | 0.50 (0.08) | 0.38 (0.05) | 0.43 (0.05) | 0.48 (0.04) |
| MIX                | 0.70 (0.15) | 0.75 (0.14) | 0.73 (0.11) | 1.10 (0.56) |
| AFLOW              | 0.58 (0.33) | 1.13 (0.28) | 0.92 (0.27) | 0.65 (0.33) |
| **Step II: With weights MEGNET** |  |  |  |  |
| Dataset1           | 0.53 (0.21) | 3.20 (2.55) | 2.14 (1.50) | 3.27 (3.96) |
| Dataset2           | 0.50 (0.08) | 0.37 (0.08) | 0.42 (0.05) | 0.49 (0.09) |
| MIX                | 0.69 (0.15) | 0.73 (0.13) | 0.71 (0.12) | 0.97 (0.20) |
| AFLOW              | 0.55 (0.38) | 1.18 (0.27) | 0.93 (0.26) | 0.61 (0.34) |
| **Step III: Retrained on AFLOW** |  |  |  |  |
| Dataset1 | 0.28 (0.10) | 1.27 (0.39) | 0.88 (0.24) | 0.66 (0.18) |
| Dataset2 | 0.56 (0.13) | 0.64 (0.21) | 0.61 (0.15) | 0.65 (0.07) |
| MIX      | 0.34 (0.13) | 0.69 (0.25) | 0.55 (0.17) | 0.65 (0.08) |


## Metric: MAE (Mean Absolute Error)

| Train on \ Test on | Dataset1 | Dataset2 | MIX | AFLOW |
|--------------------|------------------|------------------|------------------|------------------|
| **Step I: No weights MEGNET** |  |  |  |  |
| Dataset1           | 75.37 (33.64) | 101.67 (37.00) | 91.26 (29.56) | 287.70 (43.83) |
| Dataset2           | 5.95 (1.38) | 4.37 (0.84) | 5.00 (0.82) | 5.50 (0.21) |
| MIX                | 29.37 (11.13) | 25.75 (8.31) | 27.18 (8.46) | 44.50 (3.79) |
| AFLOW              | 4.25 (2.67) | 5.25 (1.42) | 4.86 (1.79) | 4.95 (1.72) |
| **Step II: With weights MEGNET** |  |  |  |  |
| Dataset1           | 79.76 (40.04) | 102.68 (37.15) | 93.61 (23.15) | 286.44 (47.86) |
| Dataset2           | 5.50 (1.01) | 3.92 (1.29) | 4.55 (0.99) | 5.34 (0.52) |
| MIX                | 31.86 (9.51) | 26.10 (7.10) | 28.38 (6.46) | 44.40 (4.05) |
| AFLOW              | 3.70 (2.37) | 4.92 (1.60) | 4.44 (1.79) | 4.58 (1.48) |
| **Step III: Retrained on AFLOW** |  |  |  |  |
| Dataset1 | 1.24 (0.47) | 4.82 (1.07) | 3.41 (0.74) | 4.35 (0.48) |
| Dataset2 | 4.57 (1.31) | 4.42 (1.58) | 4.48 (1.22) | 5.54 (0.56) |
| MIX      | 2.44 (1.95) | 4.19 (1.53) | 3.50 (1.57) | 4.95 (0.67) |


## Metric: RMSE (Mean Squared Error)

| Train on \ Test on | Dataset1 | Dataset2 | MIX | AFLOW |
|--------------------|------------------|------------------|------------------|------------------|
| **Step I: No weights MEGNET** |  |  |  |  |
| Dataset1           | 152.49 (66.44) | 191.36 (92.18) | 176.46 (61.72) | 702.37 (116.27) |
| Dataset2           | 8.22 (1.64) | 6.46 (1.22) | 7.12 (1.06) | 8.16 (0.33) |
| MIX                | 50.77 (19.10) | 44.43 (16.74) | 46.43 (14.82) | 93.11 (8.65) |
| AFLOW              | 6.06 (3.34) | 7.82 (2.10) | 7.15 (2.35) | 8.73 (2.17) |
| **Step II: With weights MEGNET** |  |  |  |  |
| Dataset1           | 162.46 (87.98) | 189.05 (92.71) | 182.12 (50.58) | 699.51 (121.79) |
| Dataset2           | 7.55 (1.42) | 5.86 (1.88) | 6.56 (1.36) | 7.92 (0.62) |
| MIX                | 55.89 (16.18) | 45.30 (15.28) | 49.12 (11.24) | 93.56 (8.88) |
| AFLOW              | 5.66 (3.75) | 7.17 (2.18) | 6.59 (2.43) | 8.15 (1.75) |
| **Step III: Retrained on AFLOW** |  |  |  |  |
| Dataset1 | 1.77 (0.74) | 7.01 (1.75) | 5.01 (1.20) | 7.86 (0.64) |
| Dataset2 | 6.89 (2.15) | 7.06 (2.45) | 6.97 (1.65) | 9.61 (0.97) |
| MIX      | 3.65 (3.13) | 6.70 (2.47) | 5.51 (2.41) | 8.72 (1.14) |


## Metric: R2 (R2 score)

| Train on \ Test on | Dataset1 | Dataset2 | MIX | AFLOW |
|--------------------|------------------|------------------|------------------|------------------|
| **Step I: No weights MEGNET** |  |  |  |  |
| Dataset1           | 0.21 (0.61) | -0.58 (0.49) | -0.22 (0.53) | -2.26 (2.65) |
| Dataset2           | -1.01 (0.43) | -0.51 (0.39) | -0.62 (0.25) | -0.80 (0.16) |
| MIX                | -0.48 (0.61) | -0.48 (0.42) | -0.40 (0.44) | -0.89 (0.60) |
| AFLOW              | -0.08 (1.11) | -0.81 (0.68) | -0.42 (0.58) | -0.14 (0.83) |
| **Step II: With weights MEGNET** |  |  |  |  |
| Dataset1           | 0.40 (0.47) | -0.51 (0.38) | -0.14 (0.33) | -1.89 (1.43) |
| Dataset2           | -0.77 (0.49) | -0.34 (0.35) | -0.49 (0.28) | -0.78 (0.37) |
| MIX                | -0.52 (0.44) | -0.51 (0.30) | -0.45 (0.24) | -0.58 (0.17) |
| AFLOW              | 0.03 (0.99) | -0.41 (0.73) | -0.17 (0.71) | -0.03 (0.83) |
| **Step III: Retrained on AFLOW** |  |  |  |  |
| Dataset1 | 0.85 (0.16) | -0.29 (0.38) | 0.19 (0.26) | 0.06 (0.47) |
| Dataset2 | -0.03 (0.36) | -0.24 (0.53) | -0.17 (0.43) | -0.28 (0.27) |
| MIX      | 0.62 (0.50) | -0.16 (0.49) | 0.17 (0.43) | -0.10 (0.25) |


### Scan over Material Project Database: 

To provide concrete validation of the best performing models, we applied them to obtain predictions for stable materials in the Material Project Database. LTC for $(BaSbO_3)_2$ (mp-9127) was then computed through robust ab-initio calculation as it was consistently found by our models to have a relatively low thermal conductivity. **The result of the computation (7.1 W/m*K) was in the same order of magnitude as the predictions from our models (1.23  W/m*K)**. This agreement underscores the model’s ability to capture critical trends in LTC prediction, even for datasets it was not directly trained on.

## Conclusions 

- Improved accuracy in predicting thermal conductivity
- Demonstrated potential for application in materials science

## How to cite ParAIsite
```txt
@misc{klochko2024transferlearningdeeplearningbased,
      title={Transfer Learning for Deep Learning-based Prediction of Lattice Thermal Conductivity}, 
      author={L. Klochko and M. d'Aquin and A. Togo and L. Chaput},
      year={2024},
      eprint={2411.18259},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2411.18259}, 
}
```
## How to use ParAIsite

Clone the project to your machine:

```bash
  git clone https://github.com/liudakl/ParAIsite.git 
```

Go to the project directory:

```bash
  cd ParAIsite
```
Please keep in mind that you need to select in the script on which dataset you would like perform training, the best architecture of MLP model, and etc. To be able reproduce the results, please keep the selections as they are. They can be changed in the *"input.json"* file. 
 
**Please, install after depending on your machine configuration and pymatgen-2023.8.10 version:** 

```bash
pip install -r requirements_cuda.txt or pip install -r requirements_cpu.txt

```
After, go to the directory, from where we do calculations:

```bash
  cd main/src
```
 
### Which data ParAIsite expects you to have before the training it?

- Requires targets (Thermal Conductivities) and inputs (structure of compounds) in pkl format as 2 separated files. They should be named as follows:
  - in the folder **structures_scalers/structures_NAME_OF_YOUR_DATASET.pkl** (contains list of structures of your data) +  **structures_scalers/NAME_OF_YOUR_DATASET.pkl** (contains list of targets with respect to structures).
- For the datasets: AFLOW, Dataset1, Dataset2, and MIX they are avaible in *structure_scaler* folder. 


### What if I do not have the data in that format ? 

- If you want to reproduce the results of the paper, no additional work needed. One can follow the steps below. 

- If you want to work with your own data, please prepare them with respect to the requirements from the subsection above. We provide an example of data preparation taken from Material Database Project (MDP):  
  - Create a *NAME_OF_YOUR_DATASET.csv* list with  material ids (mpd-ids from MDP) ; Column name should be called "TC". 
  - Create a list of targets with respect to their structures and name it *NAME_OF_YOUR_DATASET.pkl*; 
  - In the file *create_structure.py* indicate your key_api for the possibility to use material project api; 
  - run :
```bash
  python create_structure.py NAME_OF_YOUR_DATASET 1 0 
```
Here important to note that first "1" means that you want to run preparation script, and "0" refers to not preparing data for scan.
At the end of the following steps, one can have in the *structures_scalers* 2 files in pkl format that contains information needed for train. 
 
### Run the model 
 
If you would like to **reproduce training without weights** (step I), please do:

```bash
  python ParAIsite_M0.py input_config.json
```

If you would like to **reproduce training with weights** (step II), please do:

```bash
  python ParAIsite_train.py input_config.json
```
or 

```bash
  sudo python ParAIsite_train.py input_config.json
```

If you would like to reproduce training after additional train on AFLOW **(step III)**, please do:

```bash
  python ParAIsite_double_train.py input_config.json
```
or 

```bash
  sudo python ParAIsite_double_train.py input_config.json
```

Please keep in mind that you need to select in the script on which dataset you would like perform training; the best architecture of MLP model, and etc. To be able reproduce the results, please keep the selections as they are.

## How to test ParAIsite on your own Validation?

Testings ParAIsite on Data are already integrated inside the code for model training/double training. 
Please keep in mind that you need to change the script with respect on which dataset you would like perform test.

## How to predict TC with already existed models of ParAIsite from our work based on your materials?

The only 2 things are required - identification of the material (in our case it is *mpd-id*) and its structure. 

- If you want to work with your own data, please prepare them with respect to the requirements from the subsection above. We provide an example of data preparation taken from Material Database Project (MDP):  
  - Create a *NAME_OF_YOUR_DATASET.csv* list with  material ids (mpd-ids from MDP) ; 
  - In the file *create_structure.py* indicate your key_api for the possibility to use material project api; 
  - run :
```bash
  python create_structure.py NAME_OF_YOUR_DATASET 0 1
```
At the end of the following steps, one can have file "structures_scalers/NAME_OF_YOUR_DATASET.pkl" that is ready as the input for scan. 

Next, you need to specify the model that will be used in testings, path to your data, and run: 

```bash
  python scan_mdp.py for_scan.josn
```
Results will appear in the folder "results_scan/". 
## References

```txt
1. Chen, C.; Ye, W.; Zuo, Y.; Zheng, C.; Ong, S. P. Graph Networks as a Universal Machine Learning Framework for
Molecules and Crystals. Chem. Mater. 2019, 31 (9), 3564–3572. https://doi.org/10.1021/acs.chemmater.9b01294.
```





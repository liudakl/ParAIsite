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
 
**Please, install after:** 

```bash
pip install -r requirements_cuda.txt or pip install -r requirements_cpu.txt

```
Depending on your situation. After, go to the directory, from where we do calculations:

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





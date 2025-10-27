# Predictive AI Model Employing Two-Stage Transfer Learning (ParAIsite) for Lattice Thermal Conductivity


<div align="center">
  <img src="https://github.com/liudakl/fine_tuning_papers/blob/main/paper/logo.png?raw=true" width="400">
</div>


In this study, we introduce **ParAIsite**, a deep learning model designed for predicting thermal conductivity of materials. Machine learning promises to accelerate the material discovery by enabling high-throughput prediction of desirable macro-properties from atomic-level descriptors or structures. However, the limited data available about precise values of these properties have been a barrier, leading to predictive models with limited precision or ability to generalize. This is particularly true of lattice thermal conductivity (LTC): existing datasets of precise (ab initio, DFT-based) computed values are limited to a few dozen materials with little variability. Based on such datasets, we study the impact of transfer learning on both the precision and generalizability of a deep learning model (ParAIsite). We start from an existing model (MEGNet[1]) and show that improvements are obtained by fine-tuning a pretrained version of it on a different tasks. Interestingly, we also show that a much greater improvement is obtained when first fine-tuning it on a large datasets of low-quality approximations of LTC (based on the AGL model), and then applying a second phase of fine-tuning with our high-quality, smaller-scale datasets. The promising results obtained pave the way not only towards a greater ability to explore large databases in search of low thermal conductivity materials but also to methods enabling increasingly precise predictions in areas where quality data are rare. 

## 🧪 Methodology

The following steps outline the process followed to achieve the results presented in this work.

### 1. **Preprocessing Data**

We begin by preparing and integrating multiple datasets for model training and evaluation.

- **Data Cleaning & Formatting:**  
  Structures and compound data are standardized to ensure compatibility with downstream models.

- **Merging Datasets:**  
  Data from different repositories are combined and harmonized into a unified format suitable for fine-tuning.

### 📂 Datasets : 

#### **Togo15**: This dataset contains **96 materials** used in a previous prediction study [2](#ref-seko2015) involving **rocksalt**, **zincblende**, and **wurtzite** structures that could be unambiguously identified in the **Materials Project Database**.  
The lattice thermal conductivity (LTC) values are obtained using the **phono3py** software package [3](#ref-phonopy), based on YAML files available through the [PhononDB](https://github.com/atztogo/phonondb) repository.  
Obtaining predictions with low deviation from these reference values is a central motivation for this work.

#### **AFLOW AGL Dataset** : This dataset contains **5,578 materials** extracted from the **AFLOW-LIB** repository [4](#ref-calderon2015), along with their estimated thermal conductivity computed using a **quasi-harmonic Debye–Grüneisen model** [5](#ref-blanco2004,#ref-toher2014).  
This dataset serves as a large-scale, lower-fidelity training source for the first stage of transfer learning.

---

### 2. **Model Development**

- **Fine-Tuning MEGNet:**  
  A pre-trained **MEGNet** model is used as the base network. We first fine-tune it on the larger AFLOW AGL dataset to capture general trends.

- **High-Quality Refinement:**  
  A second fine-tuning stage is then performed using the high-quality Togo15 dataset to improve precision and generalizability.

- **Custom MLP Architectures:**  
  Additional multilayer perceptron (MLP) models are developed and tested to benchmark performance against the fine-tuned MEGNet.

### 🧩 Figure: Model Workflow

<p align="center">
  <img src="workflow_2.pdf" alt="Model workflow diagram" width="60%">
</p>

**Figure:** *Sketch representing the different models trained for comparison in our methodology. Training datasets are illustrated as cylinders, and the resulting models after training **ParAIsite** are represented as cubes.*

Models are labeled from left to right as follows:  
- **Step 1 (no pre-training):** Random Weights Togo15 (**RWTG15**), Random Weights AFLOW (**RWAF**).  
- **Step 2 (using pre-trained MEGNet weights on formation energy):** Formation Energy Togo15 (**FETG15**), Formation Energy AFLOW (**FEAF**).  
- **Step 3 (transfer learning on fine-tuned AFLOW model):** Formation Energy AFLOW Togo15 (**FEAFTG15**), Random Weights AFLOW Togo15 (**RWAFTG15**).
---

### 3. **Evaluation**

- **Performance Assessment:**  
  Model accuracy is evaluated using standard regression metrics (MAE, RMSE, R²).

- **Baseline Comparison:**  
  Results are compared against conventional machine learning models and previously reported methods to assess improvements from the two-stage transfer learning approach.



## 📊 Results: 

### Validation Results (Dataset on Train vs Dataset on Validation)


## Metric: MAPE (Mean Average Percentage Error) 

| **Model**       | **Togo15** | **AFLOW** |
|------------------|:----------:|:----------:|
| **Step 1**       |            |            |
| RWTG15           | 0.55 (0.20) | 2.28 (1.12) |
| RWAF             | 0.58 (0.33) | 0.65 (0.33) |
| **Step 2**       |            |            |
| FETG15           | 0.53 (0.21) | 3.27 (3.96) |
| FEAF             | 0.55 (0.38) | 0.61 (0.34) |
| **Step 3**       |            |            |
| FEAFTG15         | 0.28 (0.10) | 0.66 (0.18) |
| **Additional Step** |         |            |
| RWAFTG15         | 0.43 (0.39) | 0.83 (0.37) |

### Scan over Material Project Database: 

To provide concrete validation of the best performing models, we applied them to obtain predictions for stable materials in the Material Project Database. LTC for $(BaSbO_3)_2$ (mp-9127) was then computed through robust ab-initio calculation as it was consistently found by our models to have a relatively low thermal conductivity. **The result of the computation (7.1 W/m*K) was in the same order of magnitude as the predictions from our models (1.23  W/m*K)**. This agreement underscores the model’s ability to capture critical trends in LTC prediction, even for datasets it was not directly trained on.

## 📌 Conclusions 

- Improved accuracy in predicting thermal conductivity
- Demonstrated potential for application in materials science

## 📑 How to cite ParAIsite
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
## 🛠️ How to use ParAIsite

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

## 🧪🔍 How to test ParAIsite on your own Validation?

Testings ParAIsite on Data are already integrated inside the code for model training/double training. 
Please keep in mind that you need to change the script with respect on which dataset you would like perform test.

## 🤖🔮 How to predict TC with already existed models of ParAIsite from our work based on your materials?

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
  python scan_mdp.py for_scan.json
```
Results will appear in the folder "results_scan/". 
## 📚 References

```txt
1. Chen, C.; Ye, W.; Zuo, Y.; Zheng, C.; Ong, S. P. Graph Networks as a Universal Machine Learning Framework for
   Molecules and Crystals. Chem. Mater. 2019, 31 (9), 3564–3572. https://doi.org/10.1021/acs.chemmater.9b01294.

2. Seko, A.; Togo, A.; Hayashi, H.; Tsuda, K.; Chaput, L.; Tanaka, I. Prediction of Low-Thermal-Conductivity Compounds
   with First-Principles Anharmonic Lattice-Dynamics Calculations and Bayesian Optimization.
   Phys. Rev. Lett. 2015, 115 (20), 205901. https://doi.org/10.1103/PhysRevLett.115.205901.

3. Togo, A.; Tanaka, I. First Principles Phonon Calculations in Materials Science.
   Scr. Mater. 2015, 108, 1–5. https://doi.org/10.1016/j.scriptamat.2015.07.021.

4. Calderon, C. E.; Plata, J. J.; Toher, C.; Oses, C.; Levy, O.; Fornari, M.; Nardelli, M. B.; Curtarolo, S.
   The AFLOW Standard for High-Throughput Materials Science Calculations.
   Comput. Mater. Sci. 2015, 108, 233–238. https://doi.org/10.1016/j.commatsci.2015.07.019.

5. Blanco, M. A.; Francisco, E.; Luaña, V. GIBBS: Isothermal–Isobaric Thermodynamics of Solids from Energy Curves Using
   a Quasi-Harmonic Debye Model. Comput. Phys. Commun. 2004, 158 (1), 57–72.
   https://doi.org/10.1016/j.cpc.2003.12.001.

6. Toher, C.; Plata, J. J.; Levy, O.; de Jong, M.; Asta, M.; Nardelli, M. B.; Curtarolo, S.
   High-Throughput Computational Screening of Thermal Conductivity, Debye Temperature, and Grüneisen Parameter Using
   a Quasi-Harmonic Debye Model. Phys. Rev. B 2014, 90 (17), 174107.
   https://doi.org/10.1103/PhysRevB.90.174107.
```

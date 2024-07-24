# ParAIsite: a Fine-Tuned Neural Network model for Predicting Thermal Conductivity


<div align="center">
<img src="https://github.com/liudakl/fine_tuning_papers/blob/main/paper/ParAIsite.png?raw=true" alt="Model" width="500" height="400">
</div>


In this study, we introduce parAIsite, a deep learning model designed for predicting thermal conductivity of materials. Despite the existence of large databases, accessing extensive thermal property data, such as thermal conductivity for various compounds, remains challenging. One effective approach for achieving precise predictions on small datasets is using the framework of Transfer Learning (TL). In TL, a pre-trained machine learning (ML) model on a large dataset can be fine-tuned on a smaller dataset. In our case, we utilized an existing model pre-trained on X data, extracted the output layer, and inserted our multi-layer perceptron (MLP) model on top of it, making its output the input for our model. This innovative approach demonstrates the potential of parAIsite to significantly advance computational materials science by providing accurate thermal conductivity predictions.


## How to run ParAIsite

Clone the project to your machine:

```bash
  git clone https://github.com/liudakl/fine_tuning_papers 
```

Go to the project directory

```bash
  cd my-project
```

To run 

```bash
  python3.10 how_to.py
```


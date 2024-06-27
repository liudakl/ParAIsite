import os
import numpy as np
import pandas as pd
import torch

from model import Model
from sklearn.metrics import roc_auc_score
from kingcrab import CrabNet
from get_compute_device import get_compute_device
from tuned_model import tuned_model

compute_device = get_compute_device(prefer_last=True)

trained_crab_model = Model(CrabNet(compute_device=compute_device).to(compute_device))
trained_crab_model.load_network('trained_models_12_04_mat2vec/CritExam__Ed.pth')





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 10:27:34 2024

@author: lklochko
"""

import numpy as np
import pandas as pd
import json
import torch 
import sys 
import os
import tensorflow as tf

from megnet.models.model_changed import MEGNetModel_Embed, MEGNetModel_D4
from megnet.models import MEGNetModel
from pymatgen.ext.matproj import MPRester

model_pretrained = MEGNetModel.from_file('pretrained_models/mp-2018.6.1/formation_energy.hdf5')

embedding_layer_loaded = [i for i in model_pretrained.layers if i.name.startswith('embedding')][0]
embedding_loaded = embedding_layer_loaded.get_weights()[0]

model_new = MEGNetModel_D4(100, 2, nvocal=95, embedding_dim=16)
embedding_layer_new = [i for i, j in enumerate(model_new.layers) if j.name.startswith('atom_embedding')][0]
model_new.layers[embedding_layer_new].set_weights([embedding_loaded])
model_new.layers[embedding_layer_new].trainable = False


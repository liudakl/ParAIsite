import os
import glob
import torch
import torch.nn as nn

     
class myMLP(nn.Module):
    def __init__(self, inpuT_size, hs1, hs2, hs3,hs4, output_size):
        
        
        
        super(myMLP, self).__init__()       
          
        self.hs2 = hs2
        self.hs3 = hs3
        self.hs4 = hs4
        self.IL = nn.Linear(inpuT_size, hs1)
        self.relu1 = nn.ReLU()


        if hs2!= 0: 
          if hs3 != 0:
              if hs4 != 0:
                  self.HL2 = nn.Linear(hs1, hs2)
                  self.relu2 = nn.ReLU()
                  self.HL3 = nn.Linear(hs2, hs3)
                  self.relu3 = nn.ReLU()
                  self.HL4 = nn.Linear(hs3, hs4)
                  self.relu4 = nn.ReLU()
                  self.FL = nn.Linear(hs4, output_size)
              else:
                  self.HL2 = nn.Linear(hs1, hs2)
                  self.relu2 = nn.ReLU()
                  self.HL3 = nn.Linear(hs2, hs3)
                  self.relu3 = nn.ReLU()
                  self.FL = nn.Linear(hs3, output_size)
          else:
             self.HL = nn.Linear(hs1, hs2)
             self.relu2 = nn.ReLU()
             self.FL = nn.Linear(hs2, output_size)
        else:
          self.FL = nn.Linear(hs1, output_size)
        self.sm = nn.Softmax(dim=1)

        
    def forward(self, x):

        predV = self.IL(x)
        predV = self.relu1(predV)
        
        if self.hs2 != 0:
          if self.hs3 != 0:
              if self.hs4 != 0:
                  predV = self.HL2(predV)
                  predV = self.relu2(predV)
                  predV = self.HL3(predV)
                  predV = self.relu3(predV)
                  predV = self.HL4(predV)
                  predV = self.relu4(predV)
              else:
                  predV = self.HL2(predV)
                  predV = self.relu2(predV)
                  predV = self.HL3(predV)
                  predV = self.relu3(predV)
          else:
             predV = self.HL(predV)
             predV = self.relu2(predV)
        predV = self.FL(predV)
        return predV
 
                    

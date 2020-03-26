#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import numpy as np
from copy import deepcopy


# In[15]:


strengths = ("No human interaction in driving",
             "Prevention of car accidents",
             "Speed limit",
             "Traffic analysis and lane moving control",
             "ABS, ESP systems",
             "Automatic Parking",
             "Fully automated Traffic Regulations",
             "Rational route selection",
             "Rational fuel usage",
             "Public transport sync",
             "Automatic photos",
             "Rational time management",
            )


# In[16]:


weaknesses = ("System wear",
              "Huge computational complexity",
              "Electric dependency",
              "Computer vision weaknesses (accuracy)",
              "Huge error price",
              "Potential persnal data leakage",
              "GPS/Internet quality",
              "No relationships with non-automated cars",
              "Trolley problem",
              "Software bugs",
              "High price",
              "Man depends on vehicle"
             )


# In[17]:


opportunities = ("E-maps of road signs",
                 "Vehicle to vehicle connection",
                 "More passanger places (no driver needed)",
                 "System to prevent trafic jams",
                 "Alternative energy usage",
                 "New powerful computer vision technologies",
                 "Cars technologies with less polution",
                 "High efficency engines",
                 "High demand on cars",
                 "Goverment support of autonomous cars dev",
                )


# In[85]:


threats = ("High production price",
           "High service price",
           "Not enough qualified mechanics",
           "Bad road cover surface",
           "Not ordinary behaivor of some road users",
           "Rejection by people",
           "Hacking, confidentiality problems",
           "Increased scam interest",
           "Job abolition",
           "Inadequacy of legal system",
          )

def deambiguos(array, letter):
    indexes = list(map(lambda index: letter + str(index + 1), range(len(array))))
    return indexes


# In[86]:


strengths_letters = deambiguos(strengths, 'S')
weaknesses_letters = deambiguos(weaknesses, 'W')
opportunities_letters = deambiguos(opportunities, 'O')
threats_letters = deambiguos(threats, 'T')

ST = pd.DataFrame(index = strengths, columns = threats)
SO = pd.DataFrame(index = strengths, columns = opportunities)
WT = pd.DataFrame(index = weaknesses, columns = threats)
WO = pd.DataFrame(index = weaknesses, columns = opportunities)
print(threats_letters)
ST_letters = pd.DataFrame(index = strengths_letters, columns = threats_letters)
SO_letters = pd.DataFrame(index = strengths_letters, columns = opportunities_letters)
WT_letters = pd.DataFrame(index = weaknesses_letters, columns = threats_letters)
WO_letters = pd.DataFrame(index = weaknesses_letters, columns = opportunities_letters)


# In[87]:


ST['High production price'] =                    [0.0, 0.09, 0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.6,]
ST['High service price'] =                       [0.4, 0.3, 0.4, 0.4, 0.1, 0.2, 0.8, 0.0, 0.1, 0.0, 0.0, 0.2,]
ST['Not enough qualified mechanics'] =           [0.0, 0.5, 0.4, 0.1, 0.8, 0.0, 0.65, 0.0, 0.33, 0.0, 0.0, 0.0,]
ST['Bad road cover surface'] =                   [0.4, 0.7, 0.6, 0.45, 0.75, 0.0, 0.8, 0.5, 0.0, 0.1, 0.0, 0.3,]
ST['Not ordinary behaivor of some road users'] = [0.8, 0.9, 0.6, 1.0, 0.05, 0.0, 0.3, 0.0, 0.0, 0.5, 0.0, 0.0,]
ST['Rejection by people'] =                      [0.0, .95, 0.0, 0.8, 0.0, 0.25, .9, 0.17, 0.5, 0.8, 0.05, .84,]
ST['Hacking, confidentiality problems'] =        [.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.2, 0.0, 0.0, 0.0,]
ST['Increased scam interest'] =                  [.75, 0.2, 0.0, 0.0, 0.0, 0.12, 0.1, 0.4, 0.3, 0.0, 0.0, 0.7,]
ST['Job abolition'] =                            [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.0, 0.0, 0.7,]
ST['Inadequacy of legal system'] =               [0.0, 0.4, 0.5, 0.2, 0.32, 0.0, 0.98, 0.0, 0.0, 0.28, 0.0, 0.0,]
 
ST_letters['T1'] = [0.0, 0.09, 0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.6,]
ST_letters['T2'] = [0.4, 0.3, 0.4, 0.4, 0.1, 0.2, 0.8, 0.0, 0.1, 0.0, 0.0, 0.2,]
ST_letters['T3'] = [0.0, 0.5, 0.4, 0.1, 0.8, 0.0, 0.65, 0.0, 0.33, 0.0, 0.0, 0.0,]
ST_letters['T4'] = [0.4, 0.7, 0.6, 0.45, 0.75, 0.0, 0.8, 0.5, 0.0, 0.1, 0.0, 0.3,]
ST_letters['T5'] = [0.8, 0.9, 0.6, 1.0, 0.05, 0.0, 0.3, 0.0, 0.0, 0.5, 0.0, 0.0,]
ST_letters['T6'] = [0.0, .95, 0.0, 0.8, 0.0, 0.25, .9, 0.17, 0.5, 0.8, 0.05, .84,]
ST_letters['T7'] = [.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.2, 0.0, 0.0, 0.0,] 
ST_letters['T8'] = [.75, 0.2, 0.0, 0.0, 0.0, 0.12, 0.1, 0.4, 0.3, 0.0, 0.0, 0.7,]
ST_letters['T9'] = [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.0, 0.0, 0.7,]
ST_letters['T10'] = [0.0, 0.4, 0.5, 0.2, 0.32, 0.0, 0.98, 0.0, 0.0, 0.28, 0.0, 0.0,]
ST.style.background_gradient(cmap='Oranges', axis=None)
ST.shape
ST_letters


# In[89]:


SO["E-maps of road signs"] =                      [0.5, 0.4, 0.8, 0.9, 0.0, 0.2, 0.7, 0.9, 0.0, 0.3, 0.8, 0.0] 
SO["Vehicle to vehicle connection"] =             [0.0, 0.2, 0.4, 0.55, 0.0, 0.0, 0.25, 0.25, 0.2, 0.7, 0.0, 0.0] 
SO["More passanger places (no driver needed)"] =  [0.99, 0.8, 0.6, 0.8, 0.2, 0.8, 0.92, 0.9, 0.5, 0.93, 0.4, 0.8] 
SO["System to prevent trafic jams"] =             [0.86, 0.97, 0.9, 0.85, 0.15, 0.45, 0.85, 1.0, 0.2, 0.1, 0.0, 0.7] 
SO["Alternative energy usage"] =                  [0.0, 0.0, 0.3, 0.2, 0.0, 0.0, 0.35, 0.28, 0.95, 0.2, 0.0, 0.1] 
SO["New powerful computer vision technologies"] = [0.41, 0.0, 0.31, 0.7, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.75, 0.0] 
SO["Cars technologies with less polution"] =      [0.8, 0.0, 0.69, 0.54, 0.0, 0.2, 0.3, 0.75, 0.88, 0.0, 0.0, 0.62] 
SO["High efficency engines"] =                    [0.8, 0.0, 0.5, 0.2, 0.3, 0.0, 0.21, 0.2, 0.8, 0.0, 0.0, 0.0]
SO["High demand on cars"] =                       [0.9, 0.85, 0.1, 0.6, 0.07, 0.36, 0.28, 0.65, 0.7, 0.0, 0.42, 0.9]
SO["Goverment support of autonomous cars dev"] =  [0.65, 0.95, 0.8, 0.75, 0.0, 0.39, 1.0, 0.0, 0.5, 0.8, 0.0, 0.59]

SO_letters['O1'] = [0.5, 0.4, 0.8, 0.9, 0.0, 0.2, 0.7, 0.9, 0.0, 0.3, 0.8, 0.0] 
SO_letters['O2'] = [0.0, 0.2, 0.4, 0.55, 0.0, 0.0, 0.25, 0.25, 0.2, 0.7, 0.0, 0.0] 
SO_letters['O3'] = [0.99, 0.8, 0.6, 0.8, 0.2, 0.8, 0.92, 0.9, 0.5, 0.93, 0.4, 0.8]
SO_letters['O4'] = [0.86, 0.97, 0.9, 0.85, 0.15, 0.45, 0.85, 1.0, 0.2, 0.1, 0.0, 0.7] 
SO_letters['O5'] = [0.0, 0.0, 0.3, 0.2, 0.0, 0.0, 0.35, 0.28, 0.95, 0.2, 0.0, 0.1] 
SO_letters['O6'] = [0.41, 0.0, 0.31, 0.7, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.75, 0.0]
SO_letters['O7'] = [0.8, 0.0, 0.69, 0.54, 0.0, 0.2, 0.3, 0.75, 0.88, 0.0, 0.0, 0.62]
SO_letters['O8'] = [0.8, 0.0, 0.5, 0.2, 0.3, 0.0, 0.21, 0.2, 0.8, 0.0, 0.0, 0.0]
SO_letters['O9'] = [0.9, 0.85, 0.1, 0.6, 0.07, 0.36, 0.28, 0.65, 0.7, 0.0, 0.42, 0.9]
SO_letters['O10'] = [0.65, 0.95, 0.8, 0.75, 0.0, 0.39, 1.0, 0.0, 0.5, 0.8, 0.0, 0.59]
SO.style.background_gradient(cmap='Greens', axis=None)
SO.shape
SO_letters


# In[92]:


WT['High production price'] =                    [0.1, 0.85, 0.33, 0.4, 0.0, 0.5, 0.4, 0.0, 0.0, 0.6, 0.9, 0.0,]
WT['High service price'] =                       [0.7, 0.0, 0.4, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.35, 0.23, 0.1,]
WT['Not enough qualified mechanics'] =           [0.65, 0.0, 0.25, 0.0, 0.1, 0.0, 0.3, 0.2, 0.0, 0.6, 0.44, 0.0,]
WT['Bad road cover surface'] =                   [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.0, 0.15, 0.3, 0.1,]
WT['Not ordinary behaivor of some road users'] = [0.69, 0.9, 0.02, 0.95, 1.0, 0.0, 0.67, 0.8, 0.75, 0.6, 0.0, 0.8,]
WT['Rejection by people'] =                      [0.0, 0.0, 0.0, 0.6, 0.85, 0.45, 0.0, 0.1, 1.0, 0.34, 0.18, 0.59,]
WT['Hacking, confidentiality problems'] =        [0.0, 0.0, 0.0, 0.0, 0.1, 0.99, 0.2, 0.0, 0.35, 0.5, 0.0, 0.2,]
WT['Increased scam interest'] =                  [0.0, 0.0, 0.0, 0.1, 0.4, 0.75, 0.35, 0.0, 0.4, 0.65, 0.95, 0.15,]
WT['Job abolition'] =                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.1, 0.1,]
WT['Inadequacy of legal system'] =               [0.0, 0.0, 0.0, 0.7, 0.92, 0.65, 0.0, 0.25, 1.0, 0.35, 0.25, 0.4,]

WT_letters['T1'] = [0.1, 0.85, 0.33, 0.4, 0.0, 0.5, 0.4, 0.0, 0.0, 0.6, 0.9, 0.0,]
WT_letters['T2'] = [0.7, 0.0, 0.4, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.35, 0.23, 0.1,]
WT_letters['T3'] = [0.65, 0.0, 0.25, 0.0, 0.1, 0.0, 0.3, 0.2, 0.0, 0.6, 0.44, 0.0,]
WT_letters['T4'] = [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.0, 0.15, 0.3, 0.1,]
WT_letters['T5'] = [0.69, 0.9, 0.02, 0.95, 1.0, 0.0, 0.67, 0.8, 0.75, 0.6, 0.0, 0.8,]
WT_letters['T6'] = [0.0, 0.0, 0.0, 0.6, 0.85, 0.45, 0.0, 0.1, 1.0, 0.34, 0.18, 0.59,]
WT_letters['T7'] = [0.0, 0.0, 0.0, 0.0, 0.1, 0.99, 0.2, 0.0, 0.35, 0.5, 0.0, 0.2,]
WT_letters['T8'] = [0.0, 0.0, 0.0, 0.1, 0.4, 0.75, 0.35, 0.0, 0.4, 0.65, 0.95, 0.15,]
WT_letters['T9'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.1, 0.1,]
WT_letters['T10'] = [0.0, 0.0, 0.0, 0.7, 0.92, 0.65, 0.0, 0.25, 1.0, 0.35, 0.25, 0.4,]
WT.style.background_gradient(cmap='Reds', axis=None)
WT.shape


# In[93]:


WO["E-maps of road signs"] =                      [0.0, 0.9, 0.2, 0.0, 0.05, 0.3, 0.8, 0.1, 0.0, 0.48, 0.75, 0.0] 
WO["Vehicle to vehicle connection"] =             [0.1, 0.96, 0.3, 0.0, 0.0, 0.7, 1.0, 0.83, 0.22, 0.56, 0.8, 0.0] 
WO["More passanger places (no driver needed)"] =  [0.3, 0.7, 0.0, 0.0, 0.8, 0.0, 0.9, 0.0, 0.76, 0.7, 0.5, 0.1] 
WO["System to prevent trafic jams"] =             [0.2, 0.8, 0.08, 0.6, 0.3, 0.2, 0.7, 0.6, 0.4, 0.4, 0.83, 0.0] 
WO["Alternative energy usage"] =                  [0.7, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.13, 0.35, 0.0] 
WO["New powerful computer vision technologies"] = [0.22, 0.9, 0.65, 0.4, 0.35, 0.12, 0.0, 0.08, 0.0, 0.5, 0.4, 0.0] 
WO["Cars technologies with less polution"] =      [0.5, 0.0, 0.37, 0.0, 0.0, 0.0, 0.3, 0.0, 0.2, 0.21, 0.12, 0.0] 
WO["High efficency engines"] =                    [0.43, 0.0, 0.25, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0]
WO["High demand on cars"] =                       [0.3, 0.0, 0.5, 0.6, 0.25, 0.4, 0.15, 0.1, 0.39, 0.7, 0.9, 0.0]
WO["Goverment support of autonomous cars dev"] =  [0.0, 0.37, 0.3, 0.3, 0.9, 0.6, 0.1, 0.09, 0.99, 0.65, 0.0, 0.39]

WO_letters['O1'] = [0.0, 0.9, 0.2, 0.0, 0.05, 0.3, 0.8, 0.1, 0.0, 0.48, 0.75, 0.0] 
WO_letters['O2'] = [0.1, 0.96, 0.3, 0.0, 0.0, 0.7, 1.0, 0.83, 0.22, 0.56, 0.8, 0.0] 
WO_letters['O3'] = [0.3, 0.7, 0.0, 0.0, 0.8, 0.0, 0.9, 0.0, 0.76, 0.7, 0.5, 0.1]
WO_letters['O4'] = [0.2, 0.8, 0.08, 0.6, 0.3, 0.2, 0.7, 0.6, 0.4, 0.4, 0.83, 0.0] 
WO_letters['O5'] = [0.7, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.13, 0.35, 0.0] 
WO_letters['O6'] = [0.22, 0.9, 0.65, 0.4, 0.35, 0.12, 0.0, 0.08, 0.0, 0.5, 0.4, 0.0]
WO_letters['O7'] = [0.5, 0.0, 0.37, 0.0, 0.0, 0.0, 0.3, 0.0, 0.2, 0.21, 0.12, 0.0]
WO_letters['O8'] = [0.43, 0.0, 0.25, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0]
WO_letters['O9'] = [0.3, 0.0, 0.5, 0.6, 0.25, 0.4, 0.15, 0.1, 0.39, 0.7, 0.9, 0.0]
WO_letters['O10'] = [0.0, 0.37, 0.3, 0.3, 0.9, 0.6, 0.1, 0.09, 0.99, 0.65, 0.0, 0.39]

WO.style.background_gradient(cmap='Blues', axis=None)
WO.shape

WO_letters


# In[70]:


SWOT = pd.concat([pd.concat([ST,SO],axis = 1),pd.concat([WT,WO],axis = 1)],axis=0)
SWOT_letteres = pd.concat((pd.concat((ST_letters, SO_letters), axis=1), pd.concat((WT_letters, WO_letters), axis=1)), axis=0)


# In[71]:


SWOT_letteres


# # EXAMPLE HOW TO USE:

# In[14]:


def form_ComponentMatrix():
    swot_matrix = SWOTComponentMatrix().load("SWOT_table.html")
    so,st,wo,wt = swot_matrix.get_components()
    swot_matrix.save_html_table("example.html", fontsize=12, font = "Comic Sans", header_color="#aef500", text_align="right")


# In[14]:





# In[16]:





# In[ ]:





# In[ ]:





# In[ ]:





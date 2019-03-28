#!/usr/bin/env python
# coding: utf-8

# # Plots

# In[ ]:


import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

WorkingFolder     = './'


# ### Baseline plot
# 
# Read baseline results, order by MLA Name in the script, unify data and plot for FS and PCA dataset.

# In[ ]:


# read the CSV file as dataframe
df = pd.read_csv(os.path.join(WorkingFolder, "ML_baseline_generator.csv"))


# Create a new dataframe with the classifiers in the script order:

# In[ ]:


classifiers =['KNeighborsClassifier','GaussianNB',
              'LinearSVC', 'LogisticRegression','LinearDiscriminantAnalysis','SVC(linear)',
              'SVC(rbf)',
              'AdaBoostClassifier', 'MLPClassifier', 
              'DecisionTreeClassifier', 'RandomForestClassifier',
              'GradientBoostingClassifier', 'BaggingClassifier','XGBClassifier']


# In[ ]:


dfc = pd.DataFrame(columns=['MLA Name'])


# In[ ]:


dfc['MLA Name'] = classifiers
dfc


# In[ ]:


dfx = pd.merge(dfc, df, on=['MLA Name'])
dfx


# In[ ]:


set(dfx["MLA_dataset"])


# Replace names of datasets:

# In[ ]:


#dfx["MLA_dataset"]= dfx["MLA_dataset"].str.replace("pca0.99.o.ds_MA_NoOutliers_tr.csv",
#                                                   "Raw-PCA.ds", case = False)
dfx["MLA_dataset"]= dfx["MLA_dataset"].str.replace("pca0.99.s.ds_MA_tr.csv",
                                                   "Std-PCA.ds", case = False)
#dfx["MLA_dataset"]= dfx["MLA_dataset"].str.replace("fs-rf.o.ds_MA_NoOutliers_tr.csv",
#                                                   "Raw-FS.ds", case = False)
dfx["MLA_dataset"]= dfx["MLA_dataset"].str.replace("fs-rf.s.ds_MA_tr.csv",
                                                   "Std-FS.ds", case = False)
#dfx["MLA_dataset"]= dfx["MLA_dataset"].str.replace("o.ds_MA_NoOutliers_tr.csv",
#                                                   "Raw.ds", case = False)
dfx["MLA_dataset"]= dfx["MLA_dataset"].str.replace("s.ds_MA_tr.csv",
                                                   "Std.ds", case = False)
dfx


# Replace some column names:

# In[ ]:


dfx.rename(columns={'MLA Name':'Classifier',
                    'MLA Test AUC':'Test AUC',
                    'MLA_dataset':'Dataset'}, 
                 inplace=True)
dfx.head()


# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")


# In[ ]:


new = dfx[["Dataset","Classifier","Test AUC"]]
new.head()


# In[ ]:


currPallete = sns.color_palette("Paired",12)
sns.palplot(currPallete)


# In[ ]:


sns.set_context('poster')


# In[ ]:


# 1. Enlarge the plot
plt.figure(figsize=(10,6))
 
sns.swarmplot(x='Test AUC', 
              y='Classifier', 
              data=new, 
              hue='Dataset',
              size=10,
              palette = 'Set1')# 2. Separate points by hue) # 3. Use Pokemon palette
 
# 4. Adjust the y-axis
# plt.ylim(0, 260)
plt.xlim(0.6, 0.9)
 
# 5. Place legend to the right
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.grid(True)


# In[ ]:


# 1. Enlarge the plot
plt.figure(figsize=(10,6))
 
sns.swarmplot(x='Test AUC', 
              y='Dataset', 
              data=new, 
              hue='Classifier',
              size=10,
              palette = currPallete)# 2. Separate points by hue) # 3. Use Pokemon palette
 
# 4. Adjust the y-axis
# plt.ylim(0, 260)
plt.xlim(0.6, 0.9)
 
# 5. Place legend to the right
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.grid(True)


# In[ ]:





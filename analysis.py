
# coding: utf-8

# In[215]:

import pandas as pd
from pandas.tools.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import os

from sklearn.preprocessing import Imputer
from sklearn import cross_validation
from sklearn.cross_validation import KFold, cross_val_score, cross_val_predict

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[1815]:

directory = "/Users/mjaron/Google Drive/QMSS/translational_bioinformatics/Research Project/Data/"
variables = pd.read_csv(directory+'variables_master.csv')
independent_variables = variables[variables['ind_dependent'] == 'independent']
dependent_variables = variables[variables['ind_dependent'] == 'dependent']

## creating a basic dictionary for lookup
var_dict = {}
for count, name in enumerate(variables['name']): 
    var_dict[str(name)] = {'label':variables['label'][count],
                               'file': variables['file'][count],
                               'year':variables['years'][count],
                                'var_type': 'na',  ##could add type, i.e. range of values, code, binary, etc
                                'ind_dep': variables['ind_dependent'][count]}

## using this to find the exact variables for each file
file_var_dict = {}
for count, file_ in enumerate(variables['file']):
    temp_df = variables[variables['file'] == file_]
    var_list = temp_df['name']
    year_list = temp_df['years']
    file_var_dict[file_] = {'variables':list(var_list), 'year':list(year_list)}


## getting actual data and reading into dict                  
data_dict = {}
for file_ in os.listdir(directory+'Demographic/'):
    if '.XPT' in file_:
        temp_df = pd.read_sas(directory+'Demographic/'+file_)
        data_dict[file_[:-4]] = temp_df

        
file_abr = list(set(variables['file']))
# print data_dict.keys()


# In[1816]:

## create a filtered data dictionary where now it narrows down to varialbes I only car about plus adding in year
## Gets entire df and just select columns needed, then adds in the year for each row.

filt_data_dict = {}
for name in data_dict.keys():
    for abr in file_abr:
        if abr in name:
            temp_df = pd.DataFrame()
            columns = file_var_dict[abr]['variables']
            for i in columns:
#                 print i
                ## hard coded in this as the variable changes name slightly from 11'-14' but pretty much same
                i1 = i
                if i == 'ACD011A':
                    i1 = 'ACD010A'
                elif i == 'ACD011B':
                    i1 = 'ACD010B'
                elif i == 'DMDBORN4' or i == 'DMDBORN2':
                    i1= 'DMDBORN'
                elif i == 'HIQ011':
                    i1= 'HID010'
                elif i == 'OCD180':
                    i1= 'OCQ180'
                try:
                    temp_df[i1] = data_dict[name][i]
                except:
                    continue
                    
#             temp_df[file_+'_year'] = [name for years in range(0,len(temp_df))] ## might not be necessary
            temp_df['SEQN'] =  data_dict[name]['SEQN']
            filt_data_dict[name] = temp_df
# print filt_data_dict.keys()


# In[1819]:

# RIDRETH1
# print var_dict['RIDRETH1']


# In[1820]:

## combine years for all same files and variables, since SEQN for each year seems to be unique

## initiate dict
combinded_var_dict ={}
for i in filt_data_dict.keys():
    for abr in file_abr:
        if abr in i:
            combinded_var_dict[abr] = pd.DataFrame()

## combine all years together for each variable
dep_list = list(dependent_variables['file'])
for i in filt_data_dict.keys():
    for abr in file_abr:
        if abr in i:
            combinded_var_dict[abr] = combinded_var_dict[abr].append(filt_data_dict[i])
            if abr in dep_list:
                combinded_var_dict[abr] = combinded_var_dict[abr].dropna()
            combinded_var_dict[abr].reset_index(drop=True, inplace=True)
    


# In[1821]:

def clean(col, max_, max_val=np.nan, min_=0, min_val=np.nan, keep_na=True,               remove=False, rem_var=99, rem_rep_var=np.nan):
    '''
    input:
        column
    arguments:
        lots of different ways to clean the column, with mostly preset info
    output:
        column
    '''
    col.loc[(col > max_)] = max_val
    col.loc[(col < min_)] = min_val
    if keep_na == False:
        col = col.fillna(value=0)  
    if remove == True:
        col.loc[(col == rem_var)] = rem_rep_var
            
    return col

#         df['ACD010B'] = df['ACD010B'].where(df['ACD010B'] != 8, 1)


# In[1822]:

# print combinded_var_dict.keys()
# print combinded_var_dict['ACQ']['ACD010B']


# In[1823]:

## now clean data based on each uniqe variable
## look at each set of variables 1 by 1 and edit if needed

'''
*ACD01** = 1 or 0, make any # > 1 = na, na = 0

*DMDBORN = 1 or 0, make any # > 1 = na
DMDHHSIZ = 1-7, 7 is 7 or more
*DMDEDUC2 & DMDEDUC3 prob should be combined 
*DMDEDUC2 = 1-5 good, 6 or more = na
*DMDEDUC3 = 1-15 good, 16 or more = na
DMDHHSZA = 0-3 good
DMDHHSZB = 0-4, 4 is 4 or more
DMDHHSZE = 0-3, 3 is 3 or more
*DMDHSEDU = 1-5 good, 6 or more = na
*DMDMARTL = 1-6 good, 6 or greater *** not linear, each val is diff, prob need to split up into indiv binary columns
*INDFMIN2 = 1-15 (not 12 & 13), good
*RIAGENDR = 1-2 good *** categorical, needs to be split up
RIDAGEYR = 0-80 good

*HID010 = 1 yes , 2 no, change to 1 yes , 0 no

LBXIN = range, good 

*OCD150 = 1 is working, anything above basically not working make 0
*OCQ180 = 1 to 133
'''

for i in combinded_var_dict:
    if 'ACQ' in i:
        df = combinded_var_dict[i]
        df['ACD010A'] = clean(df['ACD010A'], max_= 1, max_val=0, keep_na=False)
        df['ACD010B'] = clean(df['ACD010B'], max_= 7, max_val=1, keep_na=False)
        combinded_var_dict[i] = df
        
    elif 'DEMO' in i:
        df = combinded_var_dict[i]
        df['DMDBORN'] = clean(df['DMDBORN'], max_= 1, max_val=0)
        df['DMDEDUC2'] = clean(df['DMDEDUC2'], max_= 6)
        df['DMDEDUC3'] = clean(df['DMDEDUC3'], max_= 16, remove=True, rem_var=14)
        df['DMDHSEDU'] = clean(df['DMDHSEDU'], max_= 5)

        ## split up below categorical variable
        df['DMDMARTL'] = clean(df['DMDMARTL'], max_= 6, keep_na=False)
        new_var_list = ['Missing', 'Married', 'Widowed', 'Divorced', 'Separated', 'Never_married', 'Living_with_partner']
        for item in range(0,7):
            ## create empty list of 0 with same len
            df['DMDMARTL_'+new_var_list[item]] = [0]*len(df['DMDMARTL']) 
            ## find the index of each item = to the number
            df['DMDMARTL_'+new_var_list[item]].loc[(df['DMDMARTL'] == item)] = 1
        ## get rid of the original column now since it has been split up    
        df.drop('DMDMARTL', axis=1, inplace=True)
        
#         new_var_list = ['Male', 'Female']
#         for item in range(1,3):
#             ## create empty list of 0 with same len
#             df['RIAGENDR_'+new_var_list[item-1]] = [0]*len(df['RIAGENDR']) 
#             ## find the index of each item = to the number
#             df['RIAGENDR_'+new_var_list[item-1]].loc[(df['RIAGENDR'] == item)] = 1
#         ## get rid of the original column now since it has been split up    
#         df.drop('RIAGENDR', axis=1, inplace=True)
        
        df['INDFMIN2'] = clean(df['INDFMIN2'], max_= 15, remove=True, rem_var=12)
        df['INDFMIN2'].loc[(df['INDFMIN2'] == 13)] = np.nan
        combinded_var_dict[i] = df
        
    elif 'HIQ' in i:
        df = combinded_var_dict[i]
        df['HID010'] = clean(df['HID010'], max_= 2, remove=True, rem_var=2, rem_rep_var=0)
        combinded_var_dict[i] = df
        
    elif 'L10AM' in i:
        continue
        
    elif 'OCQ' in i:
        df = combinded_var_dict[i]
        df['OCD150'] = clean(df['OCD150'], max_= 1, max_val=0)
        df['OCQ180'] = clean(df['OCQ180'], max_= 133)
        combinded_var_dict[i] = df


# In[1824]:

# get kernuys cleaned data and at it all together, for independent variables
new_master = pd.DataFrame() 
for files in os.listdir(directory+'cleaned_data/'):
    if '.csv' in files:
        temp_df = pd.read_csv(directory+'cleaned_data/'+files)
        temp_df.drop('Unnamed: 0', axis=1, inplace=True)
        temp_df.drop('year', axis=1, inplace=True)
        new_master = new_master.append(temp_df)
        new_master = new_master.reset_index(drop=True)

new_master['SEQN'] = new_master['seqn']
new_master.drop('seqn', axis=1, inplace=True)

## when ready add to main df
combinded_var_dict['other'] = new_master

print new_master.columns.tolist()
print new_master.shape
print combinded_var_dict.keys()

for i in new_master.columns.tolist():
    print i
    print new_master[i].describe()
    print '\n'

'''
'alq120q', 'alq130', duplicates
bpq100b, maybe to telling
imq030, maybe to telling, vaccine
'''


# In[1825]:

### get kernyus dependent variables
dependent_var_clean = pd.DataFrame() 
for files in os.listdir(directory+'cleaned_data/dependent/'):
    if '.csv' in files:
        temp_df = pd.read_csv(directory+'cleaned_data/dependent/'+files)
        temp_df.drop('Unnamed: 0', axis=1, inplace=True)
        temp_df.drop('year', axis=1, inplace=True)
        dependent_var_clean = dependent_var_clean.append(temp_df)
        dependent_var_clean = dependent_var_clean.reset_index(drop=True)

dependent_var_clean['SEQN'] = dependent_var_clean['seqn']
dependent_var_clean.drop('seqn', axis=1, inplace=True)
dependent_var_clean.drop('lbxinsi', axis=1, inplace=True)
dependent_var_clean.drop('lbxin', axis=1, inplace=True)
## add in my cleaned column
print dependent_var_clean.shape
dependent_var_clean = dependent_var_clean.merge(combinded_var_dict['L10AM'], on='SEQN', how='inner')
dependent_var_clean = dependent_var_clean.merge(combinded_var_dict['BPX'], on='SEQN', how='inner')
# temp_bp1 = dependent_var_clean
# temp_bp1 = temp_bp1.merge(combinded_var_dict['BPX'], on='SEQN', how='inner')

print combinded_var_dict['BPX'].shape, 'here'
print combinded_var_dict['L10AM'].shape
print dependent_var_clean.shape
print dependent_var_clean.head()
# print new_master.sort('SEQN', ascending=False).head()
# print combinded_var_dict['L10AM'].sort('SEQN', ascending=False).head()
## when ready add to main df
# combinded_var_dict['other'] = new_master


# In[ ]:




# In[1826]:

# prob can merge on SEQN since each year seems to be unique, and only merge all independent variables
master = combinded_var_dict['ACQ']
for df_name in combinded_var_dict: 
    if 'ACQ' not in df_name and df_name not in dep_list:
        master = master.merge(combinded_var_dict[df_name], on='SEQN', how='outer')
print master.shape
# print master.describe()
# print master['DMDHHSZA'].mean()


# In[1827]:

## merge each df with each dependent variable to now use for the model
model_data_dict = {}
for i in dependent_var_clean.columns.tolist():
    if i != 'SEQN':
        temp_dep_df = pd.DataFrame()
#         if i == 'BPXSY1' or i == 'BPXDI1' or i == 'BPXPLS':
#             temp_dep_df['SEQN'] = combinded_var_dict['BPX']['SEQN']
#             temp_dep_df[i] = combinded_var_dict['BPX'][i]
#             model_data_dict[i] = master.merge(temp_dep_df, on='SEQN', how='inner')
#             print model_data_dict[i].shape
#         else:
#             temp_dep_df['SEQN'] = dependent_var_clean['SEQN']
#             temp_dep_df[i] = dependent_var_clean[i]
#             model_data_dict[i] = master.merge(temp_dep_df, on='SEQN', how='inner')
#             print model_data_dict[i].shape
        temp_dep_df['SEQN'] = dependent_var_clean['SEQN']
        temp_dep_df[i] = dependent_var_clean[i]
        model_data_dict[i] = master.merge(temp_dep_df, on='SEQN', how='inner')
        print model_data_dict[i].shape

# for i in dependent_variables['file']:
#     print i
#     var = combinded_var_dict[i].columns.tolist()
#     print var
#     model_data_dict[var[0]] = master.merge(combinded_var_dict[i], on='SEQN', how='inner')
#     print model_data_dict[var[0]].shape
print model_data_dict.keys()


# In[1828]:

# model_data_dict['lbdldl']['DMDMARTL_Widowed'].describe()


# In[1829]:

## get rid of outliers
for i in model_data_dict:
    df = model_data_dict[i]
    for col in df.columns.tolist():
        #keep only the ones that are within +3 to -3 standard deviations in the column
        df[col].loc[np.abs(df[col]-df[col].mean()) > (3*df[col].std())] = np.nan
#         df = df[np.abs(df[col]-df[col].mean())<=(3*df[col].std())] 
    model_data_dict[i] = df
#     print df.shape()


# In[1830]:

## fill nan, using mean of entire column for now

for i in model_data_dict:
    print i
    col_length = []
    df = model_data_dict[i]
    df.reset_index(drop=True, inplace=True)
#     imp = Imputer(missing_values='NaN', strategy='mean', axis=0, verbose=0, copy=True)
#     imp.fit_transform(X)
    for col in model_data_dict[i]:
        perc = float(len(df[col].dropna()))/len(df)
        print col, len(df[col].dropna()), len(df), perc
        na_val = np.mean(df[col])
        
        ## find ~2000 rows with the least na's
#         row_count = df.count(axis=1)
#         row_count.sort_values(ascending=False, inplace=True)
#         row_count = row_count.reset_index()
#         ## get only top 2000 rows
#         row_index = list(row_count['index'][:4000])
        
        ## threshold to get rid of columns if missing to much data
        if np.isnan(na_val) or perc <= .4:
            print "dropped ", col
            df.drop(col, axis=1, inplace=True)
        else:
            df[col].fillna(na_val, inplace=True)
        ## now after replacement and threshold keep only top 2000 rows with least NA's calculated abovee
#     df = df.loc[row_index]
#     df = df.sample(n=2000)
        
    df.reset_index(drop=True, inplace=True)
    model_data_dict[i] = df

    


# In[1831]:

# temp = pd.DataFrame({'a':[1,2,3,4,5,np.nan,6], 'b':[2,3,4,6,np.nan,np.nan,9]})
# a = temp.count(axis=1)
# print a.sort_values(ascending=False, inplace=True)
# print a
# b = a.reset_index()
# k = list(b['index'][:3])
# print k
# print temp.loc[k]


# In[1832]:

## find out which variables correlate with the dependent var

## general features to remove for all
# features_to_remove = ['ACD010B', 'DMDMARTL_Widowed', 'DMDMARTL_Divorced', 'DMDMARTL_Separated', \
#                       'DMDMARTL_Living_with_partner', 'DMDMARTL_Missing']

print model_data_dict.keys()

for i in model_data_dict:
    print i
#     try:
#         model_data_dict[i] = model_data_dict[i].drop(features_to_remove, axis=1)
#     except:
#         pass
    df = model_data_dict[i]
    a = df.corr()
#     print a
#     b = a[i]
#     print b.sort(ascending=True, kind='quicksort', na_position='last', inplace=False)
    high = []
    for col in df.columns.tolist():
        b = a[col]
        b = b.reset_index()
        for count,x in enumerate(b[col]):
            if abs(x) > .2 and abs(x) != 1:
                high.append((col,b['index'][count],x))
    print '*********************'
    break
print high
# plt.scatter(model_data_dict['lbdldl']['RIAGENDR'],model_data_dict['lbdldl']['whd120'])
# a = model_data_dict['lbdldl']['lbdldl']
# b = model_data_dict['lbxsch']['lbxsch']
# c = model_data_dict['lbxstr']['lbxstr']
# print '\n'
# print np.corrcoef(a,b)
# print np.corrcoef(a,c)
# print np.corrcoef(b,c)


# In[1833]:

## correlations
'''
[('RIDAGEYR', 'whq150', 0.63273907024842002), ('bmxbmi', 'bmxwaist', 0.86370485543272124), 
('bmxbmi', 'whd020', 0.66453481733176267), ('bmxwaist', 'bmxbmi', 0.86370485543272124), 
('bmxwaist', 'whd020', 0.67986006498703533), ('whd020', 'bmxbmi', 0.66453481733176267), 
('whd020', 'bmxwaist', 0.67986006498703533), ('whd020', 'whd110', 0.56738619317411143), 
('whd020', 'whd120', 0.542019736144144), ('whd110', 'whd020', 0.56738619317411143), 
('whd110', 'whd120', 0.6215479165156752), ('whd120', 'whd020', 0.542019736144144), 
('whd120', 'whd110', 0.6215479165156752), ('whq150', 'RIDAGEYR', 0.63273907024842002)]
'''
## variables to remove
## after the 3rd row is experimentation
features_to_remove = ['DMDEDUC3', 'ACD010B', 'DMDMARTL_Widowed', 'DMDMARTL_Missing', 
                      'DMDMARTL_Divorced','DMDMARTL_Separated', 'DMDMARTL_Living_with_partner', 'hsq520', 
                      'bmxwaist', 'whd020', 'whq150', 'whd110', ## > 50% correlation with other variables
                     'DMDMARTL_Never_married', 'DMDHSEDU', 'DMDHHSIZ', 'DMDBORN'  ## > 40% correlation
                     ]

print model_data_dict.keys()

for i in model_data_dict:
    print i
    print model_data_dict[i].shape
#     print model_data_dict[i].columns.tolist()
    for var in features_to_remove:
        try:
            model_data_dict[i] = model_data_dict[i].drop(var, axis=1)
        except:
#             print "didnt drop ", var
            pass


# In[1834]:

### turn into classification model, requires more knowledge of data
'''
http://www.healthline.com/health/high-cholesterol/levels-by-age#3
lbdldl
LDL
Good: 130 mg/dL or lower  (0)
Borderline: 130 to 159 mg/dL   (1)
High: 160 mg/dL or higher   (2)

lbxsch
Total Cholesterol
Good: 200 mg/dL or lower   (0)
Borderline: 200 to 239 mg/dL   (1)
High: 240 mg/dL or higher   (2)

lbxstr (triglycerides)
Triglycerides
Good: 149 mg/dL or lower  (0)
Borderline: 150 to 199 mg/dL   (1)
High: 200 mg/dL or higher   (2)

lbxsgl (glucose)


lbxsca (calcium)


LBXIN (Insulin)
A study in Arizona found that women with a fasting insulin level around 8.0 had twice the risk of prediabetes as 
did women with a level around 5.0. Women with a fasting insulin of 25 or so had five times the risk of prediabetes.

(if to the tenth then 6.5)
2-6 good
> 7 bad

lbxsir (Iron)


'BPXDI1' (diastolic)  'BPXSY1' (systolic)
90 over 60 (90/60) or less: You may have low blood pressure
More than 90 over 60 (90/60) and less than 120 over 80 (120/80): Your blood pressure reading is ideal and healthy
More than 120 over 80 and less than 140 over 90 (120/80-140/90): You have a normal blood pressure reading but it is 
a little higher than it should be, and you should try to lower it
140 over 90 (140/90) or higher (over a number of weeks): You may have high blood pressure (hypertension).
Change your lifestyle - see your doctor or nurse and take any medicines they may give you. 
3 = high
2 = pre-high
1 = good
0 = low

'BPXPLS'  (pulse)




'''
cat_list = ['lbdldl', 'lbxsch', 'lbxstr', 'BP', 'LBXIN']
## copy data to new dict name

var = 'LBXIN'
if max(model_data_dict[var][var]) > 5:
#     plt.hist(model_data_dict[var][var])
    model_data_dict[var].reset_index(drop=True, inplace=True)
    df = model_data_dict[var]
    df[var].loc[df[var] <= 6.599] = 0
    df[var].loc[df[var] >= 6.6] = 1
    model_data_dict[var] = df
    


dia = 'BPXDI1'
sys = 'BPXSY1'
if max(model_data_dict['BPXDI1']['BPXDI1']) > 5:
    model_data_dict[dia].reset_index(drop=True, inplace=True)
    model_data_dict[sys].reset_index(drop=True, inplace=True)
    dia = model_data_dict[dia]
    sys = model_data_dict[sys]
    temp_sys = pd.DataFrame()
    temp_sys['SEQN'] = sys['SEQN']
    temp_sys['BPXSY1'] = sys['BPXSY1']
    df = dia.merge(temp_sys, on='SEQN', how='inner')
    
    BP_cat = []
    for d,s in zip(df['BPXDI1'],df['BPXSY1']):
        if s > 140 or d > 90:
            BP_cat.append(3)
        elif s <= 90 or s <= 60:
            BP_cat.append(0)
        elif s <= 140:
            if d > 80 and d <= 90:
                BP_cat.append(2)
            elif s > 120:
                BP_cat.append(2)
            elif s <= 120:
                if d > 60 and d <= 80:
                    BP_cat.append(1)
                elif s > 90:
                    BP_cat.append(1)
            else:
                BP_cat.append(0)
    df.drop(['BPXSY1','BPXDI1'], axis=1, inplace=True)
    df['BP'] = BP_cat
    model_data_dict['BP'] = df


var = 'lbdldl'
if max(model_data_dict[var][var]) > 5:
    model_data_dict[var].reset_index(drop=True, inplace=True)
    df = model_data_dict[var]
    df[var].loc[df[var] <= 129] = 0
    df[var].loc[(df[var] >= 130) & (df[var] <= 159)] = 1
    df[var].loc[df[var] >= 160] = 2

    model_data_dict[var] = df
    
    
var = 'lbxsch'
if max(model_data_dict[var][var]) > 5:
    model_data_dict[var].reset_index(drop=True, inplace=True)
    df = model_data_dict[var]
    df[var].loc[df[var] <= 200] = 0
    df[var].loc[(df[var] >= 201) & (df[var] <= 239)] = 1
    df[var].loc[df[var] >= 240] = 2

    model_data_dict[var] = df
    
var = 'lbxstr'
if max(model_data_dict[var][var]) > 5:
    model_data_dict[var].reset_index(drop=True, inplace=True)
    df = model_data_dict[var]
    df[var].loc[df[var] <= 149] = 0
    df[var].loc[(df[var] >= 150) & (df[var] <= 199)] = 1
    df[var].loc[df[var] >= 200] = 2

    model_data_dict[var] = df
    

    
    
    # print 'new'
# print df.head()
# print 'old'
# print model_data_dict['lbdldl']['lbdldl'].head()
# print model_data_dict.keys()
# print model_data_dict['lbdldl_cat']['lbdldl'].head()
# print df['lbdldl_cat'].head()
# print '*****'
# print model_data_dict['lbdldl']['lbdldl'].head()


## svm regressor or linear regressor to check
## look into outliers by lookign looking at scatterplots
## look at correlations between features
## look at correlations between X and Y, if 0 correlation take it out, easy to look at scatterplot


# In[1835]:

def regr_acc(X_validation, Y_validation, model):
    # The mean squared error
    mse = np.mean((model.predict(X_validation) - Y_validation) ** 2)
    print("Mean squared error: %.2f"
          % mse)
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % model.score(X_validation, Y_validation))
    
    return mse


# In[1836]:

# model_data_dict[dependent]['LBXIN'].describe()


# In[1837]:

# for dependent in model_data_dict:
dependent = 'BP'
feature_list = model_data_dict[dependent].drop([dependent,'SEQN'], axis=1).columns.tolist()
Y = np.array(model_data_dict[dependent][dependent])
X = np.array(model_data_dict[dependent].drop([dependent,'SEQN'], axis=1))
drop = []
# for i in range(len(feature_list)):
#     print i
validation_size = 0.40
seed = 7
X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(X, Y,                                                             test_size=validation_size, random_state=seed)
Y_validation = Y_train
X_validation = X_train

if categories:
    if dependent in cat_list:
        print dependent
        model = RandomForestClassifier(n_estimators=100, min_samples_split=2,min_samples_leaf=1,                                       n_jobs=-1, max_features='auto', random_state=seed)

        score = cross_val_score(model, X_train, Y_train, cv=10)
        print("Accuracy: %s (+/- %s)" % (score.mean(), score.std()))
        cv_predict = cross_val_predict(model, X_validation,Y_validation, cv=10)
        print accuracy_score(Y_validation, cv_predict), " CV HO"
        print pd.crosstab(Y_validation, cv_predict, rownames=['True'], colnames=['Predicted'], margins=True)
        print(classification_report(Y_validation, cv_predict))
        model.fit(X_train, Y_train) 
        impo_features = feature_importance(model, X, feature_list, dependent, show_ranking=False,                                           csv=True, show_plot=True)
        roc_curve_plot(X,Y, dependent, RF_main_model) 
#         plot_learning_curve(model, dependent, X_train, Y_train, ylim=(0.5, 1.01), cv=10, n_jobs=4)


#                 print impo_features
#             drop.append(impo_features[-1])
#             X = np.array(model_data_dict[dependent].drop([dependent,'SEQN'], axis=1))
#             feature_list = model_data_dict[dependent].drop([dependent,'SEQN'], axis=1).columns.tolist()
#             for item in drop:
#                 X = np.array(model_data_dict[dependent].drop(item, axis=1))
#                 feature_list = model_data_dict[dependent].drop(item, axis=1).columns.tolist()



# In[1838]:


## split up the data
categories = True
roc_scores = {}
RF_scores = {}
# var_dict = {}
for dependent in model_data_dict:
    print dependent
    feature_list = model_data_dict[dependent].drop([dependent,'SEQN'], axis=1).columns.tolist()
    temp_list = []
    
    Y = np.array(model_data_dict[dependent][dependent])
    X = np.array(model_data_dict[dependent].drop([dependent,'SEQN'], axis=1))
    
    validation_size = 0.4
    seed = 1
    X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(X, Y,                                                                 test_size=validation_size, random_state=seed)
#     Y_validation = Y_train
#     X_validation = X_train
    ### could do cross validation with all data
    if categories:
        if dependent in cat_list:
            print dependent
    #         print '\n'
            print "categorical"
            model = RandomForestClassifier(n_estimators=100, min_samples_split=2,min_samples_leaf=1,                                           n_jobs=-1, max_features='auto', random_state=seed)
            RF_main_model = model
            model.fit(X_train, Y_train) 
            print model.score(X_train,Y_train), 'on training'
            predicted = model.predict(X_validation)
            ho_score = model.score(X_validation, Y_validation)
            print ho_score, 'on hold out'
            
            score = cross_val_score(model, X_train, Y_train, cv=10)
            print("Accuracy: %s (+/- %s)" % (score.mean(), score.std()))
            RF_scores[dependent] = [{'HO':ho_score, "CV":score}]
#             print pd.crosstab(Y_validation, predicted, rownames=['True'], colnames=['Predicted'], margins=True)
#             print 'cv'
            cv_predict = cross_val_predict(model, X_validation,Y_validation, cv=10)
            print accuracy_score(Y_validation, cv_predict), " CV HO"
        
#             print accuracy_score(Y_validation, cv_predict, normalize=False), " CV HO, count"
#             print f1_score(Y_validation, cv_predict, average='weighted'), "F1 score"
            
#             print cross_val_score(model, X, Y, cv=10).mean(), "all the data"
#             cv_predict_all = cross_val_predict(model, X,Y, cv=10)
#             print pd.crosstab(Y, cv_predict_all, rownames=['True'], colnames=['Predicted'], margins=True)
#             print 'end all test'
            print pd.crosstab(Y_validation, cv_predict, rownames=['True'], colnames=['Predicted'], margins=True)
            print(classification_report(Y_validation, cv_predict))
            plot_learning_curve(model, dependent, X_train, Y_train, ylim=(0.5, 1.01), cv=10, n_jobs=4)
#             roc_curve_plot(X,Y, dependent, RF_main_model) 

#             print temp_list
            feature_importance(model, X, feature_list, dependent, show_ranking=True,                                                   csv=True, show_plot=True)


#     else:
#         print dependent
#         print "linear"
#         model = RandomForestRegressor()  
#     #     model = PLSRegression(n_components=1)
        
#         model.fit(X_train, Y_train) 
#         print model.score(X_train,Y_train), "R^2"
# #         prediction = model.predict(X_validation)
#         mse = regr_acc(X_validation, Y_validation, model) ## my own function for mean sqrd error and variance
#         score = cross_val_score(model, X_train, Y_train, cv=10, scoring='neg_mean_squared_error').mean()
#         print score
#         cv_predict = cross_val_predict(model, X_validation,Y_validation)
#         mse = np.mean((cv_predict- Y_validation) ** 2)
#         RF_scores[dependent] = [{'HO':mse, "CV":score}]
#         feature_importance(model, X)

#     print accuracy_score(prediction,Y_validation)

#     print type(X), type(Y)
#     # prepare models
#     models = []
#     models.append(('RF', RandomForestRegressor()))
#     models.append(('PLS', PLSRegression(n_components=1)))
#     models.append(('KR', KernelRidge(alpha=1.0)))
#     models.append(('LR', LinearRegression()))
    
    
    # evaluate each model in turn
#             num_folds = 10
#             num_instances = len(X_train)
#             seed = 7
#             scoring = 'accuracy'
#             models = []
#             models.append(('LR', LogisticRegression()))
# #             models.append(('LDA', LinearDiscriminantAnalysis()))
# #             models.append(('KNN', KNeighborsClassifier()))
# #             models.append(('CART', DecisionTreeClassifier()))
# #             models.append(('NB', GaussianNB()))
# #             models.append(('SVM', SVC()))
#             # evaluate each model in turn
#             results = []
#             names = []
#             for name, model in models:
#                 kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
#                 cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
#                 results.append(cv_results)
#                 names.append(name)
#                 msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#                 print(msg)
#                 print sigmoid( dot([val1, val2], lr.coef_) + lr.intercept_ ) 
#                 print model.fit(X_train,Y_train).predict_proba(X_validation)
#                 pca = PCA(n_components=2)
#                 cls = LogisticRegression() 
#                 pipe = Pipeline([('pca', pca), ('logistic', clf)])
#                 pipe.fit(X_train, Y_train)
#                 predictions = pipe.predict(X_validation)
#                 print accuracy_score(Y_validation, predictions)
#                 print pipe.score(X_validation, Y_validation)
                
    # boxplot algorithm comparison
#     fig = plt.figure()
#     fig.suptitle('Algorithm Comparison')
#     ax = fig.add_subplot(111)
#     plt.boxplot(results)
#     ax.set_xticklabels(names)
#     plt.show()
#     break


# In[1652]:

print RF_scores


# In[ ]:

'''
Default
lbdldl
categorical
0.718861209964 on hold out
Accuracy: 0.71 (+/- 0.00)
LR: 0.735909 (0.007158)
LDA: 0.736014 (0.006932)
KNN: 0.704350 (0.007298)
lbxsch
categorical
0.637429348964 on hold out
Accuracy: 0.63 (+/- 0.00)
LR: 0.661381 (0.008607)
LDA: 0.661329 (0.007788)
KNN: 0.630607 (0.007509)
lbxstr
categorical
0.776219384551 on hold out
Accuracy: 0.78 (+/- 0.00)
LR: 0.798972 (0.009357)
LDA: 0.798658 (0.009168)
KNN: 0.780236 (0.008511)

min_samples_leaf=50
lbdldl
categorical
0.746284278836 on hold out
Accuracy: 0.74 (+/- 0.00)
lbxsch
categorical
0.674481892401 on hold out
Accuracy: 0.66 (+/- 0.00)
lbxstr
categorical
0.790663596399 on hold out
Accuracy: 0.80 (+/- 0.00)

100 estimators
lbdldl
categorical
0.737910822692 on hold out
Accuracy: 0.73 (+/- 0.00)
lbxsch
categorical
0.653757588445 on hold out
Accuracy: 0.65 (+/- 0.00)
lbxstr
categorical
0.785639522713 on hold out
Accuracy: 0.79 (+/- 0.00)

Default
lbdldl
Predicted   0.0  1.0  2.0   All
True                           
0.0        3492   61   12  3565
1.0         805   28    3   836
2.0         362   14    0   376
All        4659  103   15  4777
lbxsch
Predicted   0.0  1.0  2.0   All
True                           
0.0        3024  175   23  3222
1.0         946  123   21  1090
2.0         394   65    6   465
All        4364  363   50  4777
lbxstr
Predicted   0.0  1.0  2.0   All
True                           
0.0        3743   21   13  3777
1.0         556    6    5   567
2.0         427    4    2   433
All        4726   31   20  4777


min leaf 100
lbdldl
categorical
Predicted   0.0   All
True                 
0.0        3565  3565
1.0         836   836
2.0         376   376
All        4777  4777
lbxsch
Predicted   0.0   All
True                 
0.0        3222  3222
1.0        1090  1090
2.0         465   465
All        4777  4777
lbxstr
Predicted   0.0   All
True                 
0.0        3777  3777
1.0         567   567
2.0         433   433
All        4777  4777


Notes:
- First istead of getting rid of outliers, I made them 'na', then afterwords computed the mean replacement
- Seems Logit might be best
- Tried manipulating RF model, with leaf sample size and estimators and other things
- tried PCA and logit, worse off
- looked at learning curve and decided to reduce the amount of NA's by only keeping only 2000 of the least NA rows
  -- get rid of outliers
  -- get rid of columns with < 40% of data
  -- do mean replacement of NA's on all columns
  
- since we dont need a lot of data changed the training size to 40% in order to have a more robust predictions score
-removed any features with > 50% correlations

- very high correlation between ldl and totchl 0.76695075 
'''


# In[1174]:

from sklearn.model_selection import validation_curve

param_range = np.logspace(-6, -1, 5)
train_scores, test_scores = validation_curve(
    RF_main_model, X, Y, param_name="gamma", param_range=param_range,
    cv=10, scoring="accuracy", n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with RF")
plt.xlabel("$\gamma$")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()


# In[1732]:

## ROC curve

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from itertools import cycle

def roc_curve_plot(X,Y, dependent, model):
    # Binarize the output
    y = label_binarize(Y, classes=[0, 1, 2])
    if dependent == 'BP':
        y = label_binarize(Y, classes=[0, 1, 2, 3])
    n_classes = y.shape[1]

    # shuffle and split training and test sets
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=.8,
                                                        random_state=0)

#     model = RandomForestClassifier()
    classifier = OneVsRestClassifier(model)
    y_score = classifier.fit(X_train, Y_train).predict_proba(X_validation)


    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_validation[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_validation.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(10,8))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    if dependent == 'BP':
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'darkgreen'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))


    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class  -'+dependent)
    plt.legend(loc="lower right")
    plt.show()

# roc_curve_plot(X,Y, dependent)   

'''
'micro':
Calculate metrics globally by counting the total true positives, false negatives and false positives.
'macro':
Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
'''


# In[1807]:

from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def plot_learning_curve(estimator, title, X, y, ylim, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    print "slope ", (test_scores_mean[0] - test_scores_mean[-1]) / (train_sizes[0] - train_sizes[-1])
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()
#     return plt

# title=dependent
# cv = ShuffleSplit(n_splits=10, test_size=0.4, random_state=0)
# plot_learning_curve(RF_main_model, title, X_train, Y_train, ylim=(0, 1.01), cv=cv, n_jobs=4)
# plt.show()


# In[ ]:

def plot12(y, predicted):
    ax = fig.add_subplots()
    ax.scatter(y, predicted)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')

    plt.show()


# In[ ]:

dependent = 'lbxsgl'
feature_list = model_data_dict[dependent].drop([dependent,'SEQN'], axis=1).columns.tolist()
Y = np.array(model_data_dict[dependent][dependent])
X = np.array(model_data_dict[dependent].drop([dependent,'SEQN'], axis=1))
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(X, Y,                                                                 test_size=validation_size, random_state=seed)
y = Y_validation
model.fit(X_train, Y_train) 
predicted = model.predict(X_validation)

print "hold out"
fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

model = RandomForestRegressor() 
# y = Y_train
# predicted = cross_val_predict(model, X_train, Y_train, cv=10)
predicted = cross_val_predict(model, X_validation, Y_validation, cv=10)


print "cross val"
fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


# In[1745]:

def feature_importance(model, X, feature_list, dependent, show_ranking=True, csv=False, show_plot=True):
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
#     print feature_list[1]
    # Print the feature ranking
    print("Feature ranking:")
    temp = []
    temp1 = []
    temp2 = []
    for f in range(X.shape[1]):
        if show_ranking:
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        temp.append(importances[indices[f]])
        temp1.append(indices[f])
        temp2.append(feature_list[indices[f]])
            
    if csv:
        df = pd.DataFrame()
        df['feature'] = temp1
        df['important_score'] = temp
        df['name'] = temp2
        
        df.to_csv(dependent+'_feature_importance.csv', index=False)

    # Plot the feature importances of the forest
    plt.figure(figsize=(15,5))
#     ax = fig.add_subplot()
#     plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
           color="orange", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    if show_plot:
        plt.show()
    
    return temp2
# feature_importance(model, X, feature_list, dependent, ranking=True, csv=False)


# In[349]:

# # Test options and evaluation metric
# num_folds = 10 # 5 or 10 does not really change
# num_instances = len(X_train)
# seed = 7
# scoring = 'accuracy'


# In[361]:

## PLS regression
PLS_scores = {}
for dependent in model_data_dict:
    print dependent
    if 'cat' in dependent:
        categories = True
        dependent = dependent[:dependent.find('_cat')]
    else:
        categories = False
    feature_list = model_data_dict[dependent].drop([dependent,'SEQN'], axis=1).columns.tolist()
#     print model_data_dict[dependent][dependent].head()
    Y = np.array(model_data_dict[dependent][dependent])
    X = np.array(model_data_dict[dependent].drop([dependent,'SEQN'], axis=1))
    
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(X, Y,                                                                 test_size=validation_size, random_state=seed)
    pls2 = PLSRegression(n_components=1)
    # pls2.fit(X_train, Y_train)
    # PLSRegression(copy=True, max_iter=500, n_components=2, scale=True,
    #         tol=1e-06)
    # Y_pred = pls2.predict(X_validation)
    x,y = pls2.fit_transform(X_train, Y_train)
    # pls2.fit(x, y)
#     print pls2.get_params(deep=True)
    mse = regr_acc(X_validation, Y_validation, pls2) 
    score = cross_val_score(pls2, X_train, Y_train, cv=10, scoring='neg_mean_squared_error').mean()
    print score
    cv_predict = cross_val_predict(pls2, X_validation,Y_validation)
    mse = np.mean((cv_predict- Y_validation) ** 2)
    PLS_scores[dependent] = [{'HO':mse, "CV":score}]


# In[362]:

print 'rf'
print RF_scores
print 'pls'
print PLS_scores


# In[100]:

## start with the models
regr = LinearRegression(fit_intercept=True, normalize=False)
# kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)

# cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
# Train the model using the training sets
regr.fit(X_train, Y_train)

predictions = regr.predict(X_validation)

m = regr.coef_
b = regr.intercept_
# print "formula: y = {0}x + {1}".format(m,b)

# print(accuracy_score(Y_validation, predictions))
## row is the thing it is actually in, in this case lots of things are thought of as 4
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(X_validation) - Y_validation) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_validation, Y_validation))

y_pred = regr.predict(X_validation)
# print y_pred
# print Y_validation
# print accuracy_score(Y_validation, y_pred)

# cv = KFold(len(X_train), 10, shuffle=True, random_state=33)

# #decf = LinearRegression.decision_function(train, target)
# test = LinearRegression.predict(X_train, Y_train)
# score = cross_val_score(regr,X_train, Y_train,cv=cv )

# print("Score: {}".format(score.mean()))


# In[102]:

from sklearn.kernel_ridge import KernelRidge
n_samples, n_features = 10, 5
clf = KernelRidge(alpha=1.0)
clf.fit(X_train, Y_train) 
# KernelRidge(alpha=1.0, coef0=1, degree=3, gamma=None, kernel='linear',
#             kernel_params=None)
regr_acc(X_validation, Y_validation, clf)
# y_kr = clf.predict(X_train)


# In[25]:

## random forest
model = RandomForestRegressor()
model.fit(X_train, Y_train) 
prediction = model.predict(X_validation)
# print prediction
print mean_squared_error(Y_validation, prediction)
regr_acc(X_validation, Y_validation, model)

# model = RandomForestRegressor(random_state=0, n_estimators=100)
score = cross_val_score(model, X_train, Y_train).mean()
print score
# regr_acc(X_validation, Y_validation, model)


## from article
# from sklearn.metrics import roc_curve, auc
# model = RandomForestClassifier(n_estimators=1000) 
# model.fit(X_train, Y_train) 
# disbursed = model.predict_proba(X_validation) 
# # fpr, tpr, _ = roc_curve(Y_validation, disbursed[:,1]) 
# # roc_auc = auc(fpr, tpr) 
# # print roc_auc
# predictions = model.predict(X_validation)
# print(accuracy_score(Y_validation, predictions))
# ## row is the thing it is actually in, in this case lots of things are thought of as 4
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))


# In[166]:

# importances = model.feature_importances_
# std = np.std([tree.feature_importances_ for tree in model.estimators_],
#              axis=0)
# indices = np.argsort(importances)[::-1]

# print feature_list
# # Print the feature ranking
# print("Feature ranking:")

# for f in range(X.shape[1]):
#     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# # Plot the feature importances of the forest
# plt.figure()
# plt.title("Feature importances")
# plt.bar(range(X.shape[1]), importances[indices],
#        color="r", yerr=std[indices], align="center")
# plt.xticks(range(X.shape[1]), indices)
# plt.xlim([-1, X.shape[1]])
# plt.show()


# In[451]:

# print("Mean squared error: %.2f"
#       % np.mean((y_kr - Y_validation) ** 2))
# Explained variance score: 1 is perfect prediction
# print('Variance score: %.2f' % clf.score(X_validation, Y_validation))


# In[390]:

# print len(X_validation), len(Y_validation)
# # Plot outputs
plt.scatter(predictions, Y_validation,  color='black')
plt.ylim(0, 100)
# plt.plot(Y_validation, predictions, color='blue',
#          linewidth=3)

# plt.xticks(())
# plt.yticks(())

# plt.show()
# # print cv_results
print max(Y_train), min(Y_train), np.mean(Y_train), np.median(Y_train), np.std(Y_train)
print len(Y_train[Y_train > 250]), len(Y_train)
# plt.hist(Y_train, bins=100, range=(0,100))
# plt.show()


# In[423]:

temp = model_data_dict[dependent]
# print temp.columns.tolist()
for i in temp.columns.tolist():
    print i
    plt.hist(temp[i])
    plt.xlabel(i)
    plt.show()


# In[245]:

print model_data_dict.keys()
# print model_data_dict['L10AM']['LBXIN'].head(90)

print len(model_data_dict['L10AM']['LBXIN'])
print len(model_data_dict['L10AM'].dropna())
# model_data_dict['L10AM'].to_csv('insulin_test.csv', index=False)


# In[30]:

## correlations
df = model_data_dict['L10AM']
x = 'INDFMIN2'
y = 'LBXIN'
plt.scatter(df[x], df[y])
# plt.ylabel(y)
# plt.xlabel(x)
a = df.corr()
# print a
b = a[y].dropna()
b.sort(ascending=True, kind='quicksort', na_position='last', inplace=False)
print b[x] #dr1tsugr + dr1ttfat + dr1tcarb + dr1tprot + dr1tkcal
print b

# print master.head()
# print master['LBXIN_y'].head()
# print filt_data_dict.keys()

# print filt_data_dict['L10AM_C_2003_2004']


# In[61]:

## if needed to change NaN to 0
# a = filt_data_dict['ACQ_B_2001_2002']
# a.fillna(value=0)

## proof that seqn are different for all years
print min(filt_data_dict['OCQ_1999_2000']['SEQN']), max(filt_data_dict['OCQ_1999_2000']['SEQN'])
print min(filt_data_dict['OCQ_B_2001_2002']['SEQN']), max(filt_data_dict['OCQ_B_2001_2002']['SEQN'])
print min(filt_data_dict['OCQ_C_2003_2004']['SEQN']), max(filt_data_dict['OCQ_C_2003_2004']['SEQN'])
print min(filt_data_dict['OCQ_D_2005_2006']['SEQN']), max(filt_data_dict['OCQ_D_2005_2006']['SEQN'])
print min(filt_data_dict['OCQ_E_2007_2008']['SEQN']), max(filt_data_dict['OCQ_E_2007_2008']['SEQN'])


# In[191]:

# data_dict.keys()


# In[47]:

combinded_var_dict['OCQ'].describe()


# In[89]:

master.head()


# In[99]:

combinded_var_dict.keys()


# In[44]:

a = filt_data_dict['DEMO_F_2009_2010']['DMDEDUC3']
print a.describe()
a = a.fillna(0)
a = list(a)
print {x:a.count(x) for x in a}


# In[409]:

var_dict['OCD150']
# 0.284399062605 DMDHHSZA 5974 24963 0.239314184994
# 0.815701372615 DMDHHSZB 5974 24963 0.239314184994
# 0.52109139605 DMDHHSZE 5974 24963 0.239314184994
# 3.33677031745 DMDHSEDU 12317 24963 0.493410247166
# 7.76969215842 INDFMIN2 11337 24963 0.454152145175


# In[1771]:

# master['OCD150'].describe()
# combinded_var_dict.keys()

a = 10
b = 5
c/=b
print c


# In[ ]:




# In[ ]:




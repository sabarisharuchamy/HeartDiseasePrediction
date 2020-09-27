import pandas as pd

df = pd.read_excel ('extention_of_Z_Alizadeh_sani_dataset.xls')
print(df)
heartdtinfo=df.select_dtypes(include=['object'], exclude=['int64','float64'])
print(heartdtinfo.info())
for x in heartdtinfo.columns:
    print(x,len(heartdtinfo[x].unique().tolist()),heartdtinfo[x].unique())
Sex_dict={'Male':1,'Fmale':2}
Obesity_dict={'Y':1,'N':2}
LAD_dict={'Stenotic':1,'Normal':2}
VHD_dict={'N':1,'mild':2,'Moderate':3,'Severe':4}
Cath_dict={'CAD':1,'Normal':2}
BBB_dict={'N':1,'LBBB':2,'RBBB':3}
df['Sex_Ordinal']=df.Sex.map(Sex_dict)
df['Obesity_Ordinal']=df.Obesity.map(Obesity_dict)
df['CRF_Ordinal']=df.CRF.map(Obesity_dict)
df['CVA_Ordinal']=df.CVA.map(Obesity_dict)
df['Airway_disease_Ordinal']=df.Airway_disease.map(Obesity_dict)
df['Thyroid_Disease_Ordinal']=df.Thyroid_Disease.map(Obesity_dict)
df['CHF_Ordinal']=df.CHF.map(Obesity_dict)
df['DLP_Ordinal']=df.DLP.map(Obesity_dict)
df['Weak_Peripheral_Pulse_Ordinal']=df.Weak_Peripheral_Pulse.map(Obesity_dict)
df['Lung_rales_Ordinal']=df.Lung_rales.map(Obesity_dict)
df['Systolic_Murmur_Ordinal']=df.Systolic_Murmur.map(Obesity_dict)
df['Diastolic_Murmur_Ordinal']=df.Diastolic_Murmur.map(Obesity_dict)
df['Dyspnea_Ordinal']=df.Dyspnea.map(Obesity_dict)
df['Atypical_Ordinal']=df.Atypical.map(Obesity_dict)
df['Nonanginal_Ordinal']=df.Nonanginal.map(Obesity_dict)
df['Exertional_CP_Ordinal']=df.Exertional_CP.map(Obesity_dict)
df['LowTH_Ang_Ordinal']=df.LowTH_Ang.map(Obesity_dict)
df['LVH_Ordinal']=df.LVH.map(Obesity_dict)
df['Poor_R_Progression_Ordinal']=df.Poor_R_Progression.map(Obesity_dict)
df['BBB_Ordinal']=df.BBB.map(BBB_dict)
df['VHD_Ordinal']=df.VHD.map(VHD_dict)
df['LAD_Ordinal']=df.LAD.map(LAD_dict)
df['LCX_Ordinal']=df.LCX.map(LAD_dict)
df['RCA_Ordinal']=df.RCA.map(LAD_dict)
df['Cath_Ordinal']=df.Cath.map(Cath_dict)
print(df)
df=df.drop(['Sex', 'Obesity', 'CRF', 'CVA', 'Airway_disease', 'Thyroid_Disease', 'CHF', 'DLP', 'Weak_Peripheral_Pulse', 'Lung_rales', 'Systolic_Murmur', 'Diastolic_Murmur', 'Dyspnea', 'Atypical', 'Nonanginal', 'Exertional_CP', 'LowTH_Ang', 'LVH', 'Poor_R_Progression', 'BBB', 'VHD', 'LAD', 'LCX', 'RCA', 'Cath'], axis = 1)
print(df)
X = df.drop('Cath_Ordinal',axis = 1)
y =df.iloc[:,-1].copy() 
#X = df.drop('Cath',axis = 1)
#y = df['Cath']
print(y)
print(X)
import numpy as np
n_samples, n_features = np.shape(X)
print(n_samples, n_features)
n_labels = np.shape(y)
print(n_labels)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
print(X_train)
print(X_test)
print(y_train)
print(y_test)
from skfeature.function.similarity_based import fisher_score  
score = fisher_score.fisher_score(X_train.values, y_train.values)
print(score)
idx = fisher_score.feature_ranking(score)
print(idx)
print(df.columns)
for i in idx:
    print(df.columns[i])

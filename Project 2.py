import pandas as pd
import numpy as np
import joblib as jb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#step1

data = pd.read_csv(r"C:\Users\solan\OneDrive\Desktop\thonny\breastcancerdataset.csv")


# Map 'M' to 1 and 'B' to 0
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
#step2

#step3
#Data Analysis
print(data.head())
print(data.shape)
print(data.info())
print(data.columns)

# Map 'M' to 1 and 'B' to 0
# data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})


x = data[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean',
      'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
      'concave points_se', 'symmetry_se', 'fractal_dimension_se','radius_worst', 'texture_worst','perimeter_worst', 'area_worst', 'smoothness_worst',
      'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']]
y = data[['diagnosis']]

print(x)
print(y)

#print(data.columns.tolist())

# Map 'M' to 1 and 'B' to 0
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})


# STEP4
# MODEL SELECTION
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.linear_model
#from sklearn.model_selection
#import LinearRegression # Lasso, Ridge, KNeighborsClassifier
#model = LinearRegression()
#model = Lasso()
model = KNeighborsClassifier(n_neighbors=1)
print("Model Selected Successfully")

#STEP5
#MODEL TRAINING

# print(x.dtypes)  # This shows datatype of each column
# print(x.select_dtypes(include=['object']))  # Shows non-numeric columns

model.fit(x,y)
print("Model trained")


# STEP6
#RESULT
y_predicted= model.predict(x)
print("ML Predicted y : ",y_predicted)

# STEP7
#MODEL ACCURACY
from sklearn.metrics import r2_score

acc = r2_score(y,y_predicted)
print("Model Performance : ",acc)

#max value of k

print(data['diagnosis'].value_counts())

# test the data through sample value

columns = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean',
      'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
      'concave points_se', 'symmetry_se', 'fractal_dimension_se','radius_worst', 'texture_worst','perimeter_worst', 'area_worst', 'smoothness_worst',
      'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']

sample_data = pd.DataFrame([[17.99,10.38,122.8,1001,0.1184,0.2778,0,3001,0.1471,0.2419,0.0787,1.0953,8.589,153.4,
                             0.0064,0.049,0.0537,0.0159,0.03,0.0062,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,
                             0.2654,0.4601,0.1189]], columns = columns)
print("diagonsis",model.predict(sample_data))




# Apply the data mining steps for the below dataset. Consider the following
# 1. Model to be considered - Linear regression
# 2. dependant variable - mpg

import pandas as pd
import pandas_profiling as pp
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression #import model
from sklearn.metrics import mean_squared_error,mean_absolute_error


df = pd.read_csv("auto-mpg.csv")

# Evaluating Dataset
print("\n\n--------------------------------------------------------------------------\n\n")
print("Dataset Head\n")
print(df.head)
print("\n\n")
print("Dataset Shape: ", df.shape)
print("\n\nDataset Info")
print(df.info())
print("\n\nThere are 2 columns that are not numerical.")
print("\nThe horsepower column contains numerics as strings.The object data type needs to be changed to float.\n\n")
print("\n\n--------------------------------------------------------------------------\n\n")


# Column Details
print(df.describe(include="all"))
print("\n\n--------------------------------------------------------------------------\n\n")


# Remove Duplicates 
df_1 = df.drop_duplicates(subset=None, keep="first",inplace=False)
print(df_1.info())
print("\nThe number of records in the new dataframe is the same as the main dataframe, so there were no duplicates found")
print("\n\n--------------------------------------------------------------------------\n\n")


# Pandas Profiling on Initial Dataset
# profile = pp.ProfileReport(df)
# profile.to_file ("Report.html")


# Convert horsepower to a numerical column
df_1['cylinders'] = pd.to_numeric(df_1['cylinders'], errors='coerce')
df_1['horsepower'] = pd.to_numeric(df_1['horsepower'], errors='coerce')
print(df_1.info())
print("\nConverting horsepower to float64 datatype has created nulls that need to be filled")
print("\n\n--------------------------------------------------------------------------\n\n")

# Horsepower Mean
horse_mean = df_1["horsepower"].mean()
print("The average value of horsepower is : ",horse_mean)
print("\n\n--------------------------------------------------------------------------\n\n")
# Fill Horsepower nulls with mean 
df_1["horsepower"] = df_1["horsepower"].fillna(horse_mean)
print(df_1.info())
print("\nAll columns now contain 398 records, so nulls have been filled")
print("\n\n--------------------------------------------------------------------------\n\n")

# Outlier Treatment
# Calculate quartiles, IQR and statistical max and min
Q1 = df_1["mpg"].quantile(0.25)
Q3 = df_1["mpg"].quantile(0.75)
IQR = Q3-Q1
st_max = Q3+(1.5*IQR)
st_min = Q1-(1.5*IQR)

# Create subset dataframe of those in the correct range
df_2_outlier_n = df_1[(df_1["mpg"]>st_min) & (df_1["mpg"]<st_max)]
print("Outlier Dataset Shape: ", df_2_outlier_n.shape) #checks how many rows will be discarded
print("1 row will be discarded as an outlier")
print("\n\n--------------------------------------------------------------------------\n\n")
df_1 = df_1[df_1["mpg"]<st_max]
df_1 = df_1[df_1["mpg"]>st_min] #repeat process on main dataframe, cross check with record count
print(df_1.info())
print(df_1.head())
print("\n\n--------------------------------------------------------------------------\n\n")

# Copy the dataframe for model comparison later
df_3 = df_1.drop(["car name"],axis=1)
print("\n\n--------------------------------------------------------------------------\n\n")
print("New dataframe info:", df_3.info())
print("\n\n--------------------------------------------------------------------------\n\n")

# Split the name column so that only the car manufacturer name is included and model is dropped
df_1["car name"] = df_1["car name"].apply(lambda x: (str(x).split(" ")[0]))
print("After lambda function: \n")
print(df_1.head())
print("\n\n--------------------------------------------------------------------------\n\n")

# print("Unique Car Names Before Name Correction: ", df_1['car name'].unique())
# Correct typos in car names
df_1['car name'] = df_1['car name'].apply(lambda x: str(x).replace('chevroelt', 'chevrolet') if 'chevroelt' in str(x) else str(x))
df_1['car name'] = df_1['car name'].apply(lambda x: str(x).replace('chevy', 'chevrolet') if 'chevy' in str(x) else str(x))
df_1['car name'] = df_1['car name'].apply(lambda x: str(x).replace('maxda', 'mazda') if 'maxda' in str(x) else str(x))
df_1['car name'] = df_1['car name'].apply(lambda x: str(x).replace('-benz', '') if 'mercedes-benz' in str(x) else str(x))
df_1['car name'] = df_1['car name'].apply(lambda x: str(x).replace('toyouta', 'toyota') if 'toyouta' in str(x) else str(x))
df_1['car name'] = df_1['car name'].apply(lambda x: str(x).replace('vokswagen', 'volkswagen') if 'vokswagen' in str(x) else str(x))
df_1['car name'] = df_1['car name'].apply(lambda x: str(x).replace('vw', 'volkswagen') if 'vw' in str(x) else str(x))

print("Unique Car Names After Name Correction: ", df_1['car name'].unique())
print("\n\n")
print(df_1.info())
print(df_1.describe())


# EVALUATION 1
print("\n\n--------------------------------------------------------------------------\n\n")
print("Evaluating model with car names included")
print("\n\n--------------------------------------------------------------------------\n\n")

# Convert categorical data into numerical by creating dummy variables
df_1 = pd.get_dummies(df_1,columns=["car name"])
print(df_1.info())
print("\n\n")
# Define selling price as dependent variable, and remainder as independent
y1 = df_1["mpg"]
X1 = df_1.drop("mpg",axis=1)
X_train,X_test,y_train,y_test = train_test_split(X1,y1,train_size=0.7,random_state=0) #if run code multiple times, same datasets will be used for each
model = LinearRegression()
model.fit(X_train,y_train)
print("\nModel Intercept: ", model.intercept_)
print("\nModel Coefficients: ", model.coef_)

# Calculate Root mean square error for evaluation metrics
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse) #use square root function from math
mae = mean_absolute_error((y_test), y_pred)
print("\nRMSE with car manufacturer name is: ",rmse)
print("\nMAE with car manufacturer name is: ",mae)

# Test Value
X_TEST1 = [[4,200,200,4000,20,80,200,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
Y_PRED1 = model.predict(X_TEST1)
print("Test Value Prediction by Model 1: ", Y_PRED1)

# EVALUATION 2
print("\n\n--------------------------------------------------------------------------\n\n")
print("Evaluating model with car names excluded")
print("\n\n--------------------------------------------------------------------------\n\n")

# Run the model on the datafram without car name included
y2 = df_3["mpg"]
X2 = df_3.drop("mpg",axis=1)
print(df_3.info())
X_train2,X_test2,y_train2,y_test2 = train_test_split(X2,y2,train_size=0.7,random_state=0) #if run code multiple times, same datasets will be used for each
model2 = LinearRegression()
model2.fit(X_train2,y_train2)
print("\nModel Intercept: ", model2.intercept_)
print("\nModel Coefficients: ", model2.coef_)

# Calculate Root mean square error for evaluation metrics
y_pred2 = model2.predict(X_test2)
mse_2 = mean_squared_error(y_test2, y_pred2)
rmse_2 = math.sqrt(mse_2) #use square root function from math
mae_2 = mean_absolute_error((y_test2), y_pred2)
print("\nRMSE without car manufacturer name is: ",rmse_2)
print("\nMAE without car manufacturer name is: ", mae)

# Test Value
X_TEST2 = [[4,200,200,4000,20,80,200]]
Y_PRED2 = model2.predict(X_TEST2)
print("\nTest Value Prediction by Model 2: ", Y_PRED2)


# CONCLUSION
print("\n\n--------------------------------------------------------------------------")
print("\n\nThe model is more accurate when car name is excluded.")
print("\n\n--------------------------------------------------------------------------")


# FINAL PANDAS PROFILING REPORT
# profile_new = pp.ProfileReport(df_3)
# profile_new.to_file ("Auto_mpg_profiling_report.html")



# Apply the data mining steps for the below dataset. Consider the following
# 1. Model to be considered - Linear regression
# 2. dependant variable - mpg

import pandas as pd
import pandas_profiling as pp
df = pd.read_csv("auto-mpg.csv")

#check that the data has been imported and some basic information about it
print(df.head)
print(df.shape)
print(df.info())
print("There are 2 columns that are not numerical")
print("the horsepower column is an object and should be a numerical column")

#print details of all columns as some are object
print("\n\n--------------------------------------------------------------------------")
print(df.describe(include="all"))
print("\n\n--------------------------------------------------------------------------")

#create a new df to contain only unique values (remove duplicates)
df_1 = df.drop_duplicates(subset=None, keep="first",inplace=False)
print(df_1.info())
print("\n\n--------------------------------------------------------------------------")
print("The number of records in the new dataframe is the same as the main dataframe, so there were no duplicates found")
print("\n\n--------------------------------------------------------------------------")

#run pandas profile report to see what other data insights can be gained
#profile = pp.ProfileReport(df)
#profile.to_file ("Report.html")

#convert horsepower to a numerical column
df_1['horsepower'] = pd.to_numeric(df_1['horsepower'], errors='coerce')
print(df_1.info())
print("\n\n--------------------------------------------------------------------------")
print("Converting horsepower to a number has created blanks which need filling")



#calculate the mean horsepower value
horse_mean = df_1["horsepower"].mean()
print("The average horsepower is :",horse_mean)
print("\n\n--------------------------------------------------------------------------")
#fill blank values with mean values
df_1["horsepower"] = df_1["horsepower"].fillna(horse_mean)
print(df_1.info())
print("All columns now contain 398 records, so blanks have been filled")
print("\n\n--------------------------------------------------------------------------")

#outlier treatment
#first calculate quartiles, IQR and statistical max and min
Q1 = df_1["mpg"].quantile(0.25)
Q3 = df_1["mpg"].quantile(0.75)
IQR = Q3-Q1
st_max = Q3+(1.5*IQR)
st_min = Q1-(1.5*IQR)
#create subset dataframe of those in the correct range
df_2_outlier_n = df_1[(df_1["mpg"]>st_min) & (df_1["mpg"]<st_max)]
print(df_2_outlier_n.shape) #checks how many rows will be discarded
print("1 row will be discarded as an outlier")
print("\n\n--------------------------------------------------------------------------")
df_1 = df_1[df_1["mpg"]<st_max]
df_1 = df_1[df_1["mpg"]>st_min] #repeat process on main dataframe, cross check with record count
print(df_1.info())
print(df_1.head())
print("\n\n--------------------------------------------------------------------------")

#copy the dataframe for model comparison later
df_3 = df_1.drop(["car name"],axis=1)
print("\n\n--------------------------------------------------------------------------")
print("\n\n--------------------------------------------------------------------------")
print("New dataframe info:", df_3.info())
print("\n\n--------------------------------------------------------------------------")

#split the name column so that only the car manufacturer name is included and model is dropped
df_1["car name"] = df_1["car name"].apply(lambda x: (str(x).split(" ")[0]))
print("after lambda function")
print(df_1.head())
print("\n\n--------------------------------------------------------------------------")

#correct typos in car names
df_1['car name'] = df_1['car name'].apply(lambda x: str(x).replace('chevroelt', 'chevrolet') if 'chevroelt' in str(x) else str(x))
df_1['car name'] = df_1['car name'].apply(lambda x: str(x).replace('chevy', 'chevrolet') if 'chevy' in str(x) else str(x))
df_1['car name'] = df_1['car name'].apply(lambda x: str(x).replace('maxda', 'mazda') if 'maxda' in str(x) else str(x))
df_1['car name'] = df_1['car name'].apply(lambda x: str(x).replace('-benz', '') if 'mercedes-benz' in str(x) else str(x))
df_1['car name'] = df_1['car name'].apply(lambda x: str(x).replace('toyouta', 'toyota') if 'toyouta' in str(x) else str(x))
df_1['car name'] = df_1['car name'].apply(lambda x: str(x).replace('vokswagen', 'volkswagen') if 'vokswagen' in str(x) else str(x))
df_1['car name'] = df_1['car name'].apply(lambda x: str(x).replace('vw', 'volkswagen') if 'vw' in str(x) else str(x))

#convert categorical data into numerical by creating dummy variables
df_1 = pd.get_dummies(df_1,columns=["car name"])
print(df_1.info())
print("\n\n--------------------------------------------------------------------------")

#split data into 2 for building as testing
#check column names
#'mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration',
#      'model_year', 'origin', 'car name_amc', 'car name_audi',
#      'car name_bmw', 'car name_buick',
#      'car name_cadillac', 'car name_capri',
#      'car name_chevrolet', 'car name_chrysler',
#      'car name_datsun', 'car name_dodge',
#      'car name_fiat', 'car name_ford', 'car name_hi', 'car name_honda',
#      'car name_mazda', 'car name_mercedes', 'car name_mercury'
#      'car name_nissan',  'car name_oldsmobile',  'car name_opel', 'car name_peugeot', 'car name_plymouth'
#      'car name_pontiac', 'car name_renault', car name_saab,  'car name_subaru',  'car name_toyota'',
#      'car name_triumph', 'car name_volkswagen', 'car name_volvo'



#define selling price as dependent variable, and remainder as independent
y = df_1["mpg"]
X = df_1.drop("mpg",axis=1)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7,random_state=None) #if run code multiple times, same datasets will be used for each
from sklearn.linear_model import LinearRegression #import model
model = LinearRegression()
model.fit(X_train,y_train)
print(model.intercept_)
print(model.coef_)

#calculate Root mean square error for evaluation metrics
y_pred = model.predict(X_test)
from sklearn.metrics import mean_squared_error,mean_absolute_error
mse = mean_squared_error(y_test, y_pred)
import math
rmse = math.sqrt(mse) #use square root function from math
mae = mean_absolute_error((y_test), y_pred)
print("RMSE is: ",rmse)
print("MAE is: ",mae)

#run the model on the datafram without car name included
y = df_3["mpg"]
X = df_3.drop("mpg",axis=1)
from sklearn.model_selection import train_test_split
X_train2,X_test2,y_train2,y_test2 = train_test_split(X,y,train_size=0.7,random_state=None) #if run code multiple times, same datasets will be used for each
from sklearn.linear_model import LinearRegression #import model
model2 = LinearRegression()
model2.fit(X_train2,y_train2)
print(model2.intercept_)
print(model2.coef_)

#calculate Root mean square error for evaluation metrics
y_pred2 = model2.predict(X_test2)
from sklearn.metrics import mean_squared_error,mean_absolute_error
mse_2 = mean_squared_error(y_test2, y_pred2)
import math
rmse_2 = math.sqrt(mse_2) #use square root function from math
mae_2 = mean_absolute_error((y_test2), y_pred2)
print("RMSE without car manufacturer name is: ",rmse_2)
print("MAE without car manufacturer name is: ",mae)

print("\n\n--------------------------------------------------------------------------")
print("The model is more accurate when car name is excluded")
print("\n\n--------------------------------------------------------------------------")




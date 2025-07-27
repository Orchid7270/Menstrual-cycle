import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from skyfield.api import load # for moon distance
import pytz
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('E:/AI/Machine learning/Own projects/Menstrual cycle/menstrual_cycle_dataset_with_factors.csv')

#1st step :Understanding data

print(df.head(3))

print(df.info()) #check data types 

print(df.isnull().sum()) #check for missing values 

print(df.describe())

print(df.sample(3))

'''Parsing date time objects using builtin functions'''

df['cycle_start'] = pd.to_datetime(df['Cycle Start Date']).dt.normalize() #to extract only year without changing the format since it has time also since
#moon distance function needs like that
df['next_cycle']=pd.to_datetime(df['Next Cycle Start Date']).dt.normalize()

df['start_year']=pd.to_datetime(df['cycle_start']).dt.year
df['start_month']=pd.to_datetime(df['cycle_start']).dt.month
df['start_day']=pd.to_datetime(df['cycle_start']).dt.day
df['predic_year']=pd.to_datetime(df['next_cycle']).dt.year
df['predic_month']=pd.to_datetime(df['next_cycle']).dt.month
df['predic_date']=pd.to_datetime(df['next_cycle']).dt.day

 #Load timescale and high-precision ephemeris
ts = load.timescale()
planets = load('de430t.bsp')  # High-precision ephemeris

# Set timezone to UTC
utc = pytz.UTC

# Function to calculate moon distance
def calculate_moon_distance(date):
    date_utc = date.replace(tzinfo=utc)  # Set timezone to UTC
    t = ts.from_datetime(date_utc)
    earth = planets['earth']
    moon = planets['moon']
    astrometric = earth.at(t).observe(moon)
    distance = astrometric.distance().km
    return distance

#Function for moon phase
def calculate_moon_phase(date):
    date_utc = date.replace(tzinfo=utc)
    t = ts.from_datetime(date_utc)
    earth = planets['earth']
    moon = planets['moon']
    sun = planets['sun']

    astrometric_moon = earth.at(t).observe(moon).apparent()
    astrometric_sun = earth.at(t).observe(sun).apparent()

    moon_elongation = astrometric_sun.separation_from(astrometric_moon)
    phase = (1 + np.cos(moon_elongation.radians)) / 2

    return phase

#for moonphase
def get_moon_phase_name(phase):
    if phase < 0.06:
        return "New Moon"
    elif phase < 0.19:
        return "Waxing Crescent"
    elif phase < 0.31:
        return "First Quarter"
    elif phase < 0.44:
        return "Waxing Gibbous"
    elif phase < 0.56:
        return "Full Moon"
    elif phase < 0.69:
        return "Waning Gibbous"
    elif phase < 0.81:
        return "Last Quarter"
    elif phase < 0.94:
        return "Waning Crescent"
    else:
        return "New Moon"
    
df['moon_distance_start'] = df['cycle_start'].apply(calculate_moon_distance)# for moon distance
df['moon_phase_value'] = df['cycle_start'].apply(calculate_moon_phase) #moon phase value
df['moon_phase'] = df['moon_phase_value'].apply(get_moon_phase_name)#moon phase
df['moon_distance_next']=df['next_cycle'].apply(calculate_moon_distance)

#Data analysis

'''numerical vs numerical'''

# plt.figure(figsize=(6,6))


# sns.scatterplot(data=df,x='moon_distance_start',y='start_day',hue='Symptoms',size='Stress Level',style='Period Length')

# # sns.pairplot(data=df, vars=['moon_distance_start', 'cycle_start', 'Age', 'BMI', 'Stress Level', 'Period Length', 'Cycle Length'], hue='Symptoms')
plt.show()

 
fig, axs = plt.subplots(2, 3, figsize=(18, 12))

# Plot each scatter plot on a separate subplot .Numerical vs numerical

sns.scatterplot(data=df, x='moon_distance_start', y='cycle_start', hue='Symptoms', size='Age', style='Period Length', ax=axs[0, 0])
sns.scatterplot(data=df, x='moon_distance_start', y='Age', ax=axs[0, 1])
sns.scatterplot(data=df, x='moon_distance_start', y='BMI', ax=axs[0, 2])
sns.scatterplot(data=df, x='moon_distance_start', y='Stress Level', ax=axs[1, 0])
sns.scatterplot(data=df, x='moon_distance_start', y='Period Length', ax=axs[1, 1])
sns.scatterplot(data=df, x='moon_distance_start', y='Cycle Length', ax=axs[1, 2])

fig.tight_layout()

plt.show()

#numerical vs categorical

fig,ax=plt.subplots(2,2,figsize=(18,12))
sns.barplot(data=df,x='Symptoms',y='Age',ax=ax[0,0])
sns.barplot(data=df,x='Symptoms',y='BMI',ax=ax[0,1])
sns.barplot(data=df,x='Symptoms',y='Stress Level',ax=ax[1,0])
sns.barplot(data=df,x='Symptoms',y='Stress Level',ax=ax[1,1])

fig.tight_layout()
plt.show()

'''Categorical vs  Categorical'''

# To check how many different type of data is available
df['Cycle Length'].value_counts().plot(kind='bar')
plt.show()


sns.barplot(data=df,x='moon_phase',y='start_day')
plt.show()

ct= pd.crosstab(df['moon_phase'],df['Symptoms']) #use pd,crosstab fuction to combine 2 dataframes

sns.heatmap(ct,annot=True,cmap='Blues')
ct.plot(kind='bar')
plt.show()



X=df.drop(['predic_date','User ID','Next Cycle Start Date', 'Cycle Start Date','cycle_start','next_cycle','moon_phase_value','Age','BMI','Stress Level','Exercise Frequency','moon_phase', 'Period Length'],axis=1)
y=df['predic_date']

#To remove highly co_related columns means they are linearly realted (use,columns in the end for getting column names)
co_relation=X.select_dtypes(exclude=['object'])

print(co_relation.corr())

#Train_test split always before preprocessing for avoiding data leakage

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
scaler=StandardScaler()

X_train[['moon_distance_start','moon_distance_next']]=scaler.fit_transform(X_train[['moon_distance_start','moon_distance_next']])
X_test[['moon_distance_start','moon_distance_next']]=scaler.transform(X_test[['moon_distance_start','moon_distance_next']])

Ohe=OneHotEncoder(sparse_output=False,drop='first') #Encoding categorical data

X_train_t=Ohe.fit_transform(X_train[['Symptoms','Diet']])#choosing categorical cols since ohe doesnt auto fit
X_test_t=Ohe.transform(X_test[['Symptoms','Diet']])

X_train_p=pd.DataFrame(X_train_t,columns=Ohe.get_feature_names_out())#Assigning features names since it get transformed to array
X_test_p=pd.DataFrame(X_test_t,columns=Ohe.get_feature_names_out())
#concatanating after transformation
X_train_f = pd.concat([X_train.drop(['Symptoms','Diet'], axis=1).reset_index(drop=True), X_train_p], axis=1)#reset index is must or else error

X_test_f = pd.concat([X_test.drop(['Symptoms','Diet'], axis=1).reset_index(drop=True), X_test_p], axis=1)


'''building the models for prediction'''

# print(X_train_f['moon_phase'])
# print(df.shape)

# 1.Linear Regression
M1=LinearRegression()
M1.fit(X_train_f,y_train)
pred_M1=M1.predict(X_test_f)
print(f'error is',mean_squared_error(y_test,pred_M1))
print(r2_score(y_test,pred_M1))

# #Cross validation

score=cross_val_score(M1,X_test_f,y_test,cv=5)

print(f'Validation score is',score.mean())

M4=Ridge(alpha=0.00001)
M4.fit(X_train_f,y_train)
pred_M4=M4.predict(X_test_f)
print(f'r2score is',r2_score(y_test,pred_M4))
print(f'Error is',mean_squared_error(y_test,pred_M4))


# co=M4.coef_
# features=M4.feature_names_in_

# weights=dict(zip(features,co))
# print(weights)

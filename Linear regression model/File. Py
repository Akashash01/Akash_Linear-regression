#import library
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm

#load data 
import os
os.chdir('C:\\Users\\NEW\\Desktop\\python\\datasets')
data = pd.read_csv('data1.csv')
pd.set_option('display.max_columns', None)

#pre-process(data wrangling)
data=data.drop(ash.columns[[0,6,7]],axis=1)
ash

#remove boxplot
data.boxplot(column=["Item_Outlet_Sales"])
def ash (data,age):
 Q1 = data[age].quantile(0.25)
 Q3 = data[age].quantile(0.75)
 IQR = Q3 - Q1
 data= data.loc[~((data[age] < (Q1 - 1.5 * IQR)) | (data[age] > (Q3 + 1.5 * IQR))),]
 return data
data = pintu(data,"Item_Outlet_Sales")

#missing values
data.isnull().sum()#missing values per variable
data = data.dropna()

#dummy variables
ak = data.loc[:,["Item_Fat_Content","Item_Type","Outlet_Size","Outlet_Location_Type","Outlet_Type"]]
ak
data = data.drop(["Item_Fat_Content","Item_Type","Outlet_Size","Outlet_Location_Type","Outlet_Type"],axis=1)
data
dum = pd.get_dummies(ak.astype(str),drop_first=True)
dum
data = pd.concat([data,dum],axis=1)
data

#linear regression model
rock=sm.ols(formula=
"Q('Item_Outlet_Sales') ~Q('Item_MRP')+Q('Item_Type_Seafood')+Q('Outlet_Location_Type_Tier 2')+Q('Outlet_Type_Supermarket Type2')",
data=ash).fit()
rock.summary()

#prediction
data["pred"] = rock.predict()
data
var = pd.DataFrame(round(rock.pvalues,3))# shows p value
rock.rsquared
var["coeff"] = rock.params

#VIF(value<2)
from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = rock.model.exog #.if I had saved data as rock
# this it would have looked like rock.model.exog
vif = [variance_inflation_factor(variables, i) for i in range(variables.shape[1])]
vif 
var["vif"] = vif
var

#mape(evaluation)
from scipy import stats
stats.shapiro(rock.resid)
data["mp"] = abs((data["Item_Outlet_Sales"] - data["pred"])/data["Item_Outlet_Sales"])
(data.mp.mean())*100#mape

#evaluation
NORMALITY TEST
from scipy import stats
stats.shapiro(rock.resid)#2nd value is p value;
from scipy.stats import normaltest
normaltest(rock.resid)

AUTO CORRELATION TEST
from statsmodels.stats import diagnostic as diag
diag.acorr_ljungbox(rock.resid , lags = 1)#2nd value is p value; 
import statsmodels.stats.api as sms

HETEROSCEDASTICITY TEST
from statsmodels.compat import lzip
name = ['F statistic', 'p-value']
test = sms.het_goldfeldquandt(rock.resid, rock.model.exog)
lzip(name, test)

#EXPORT DATA
os.chdir('C:\\Users\\NEW\\Desktop\\python\\datasets')
df.to_csv('prediction.csv',index=False)#export data to local drive

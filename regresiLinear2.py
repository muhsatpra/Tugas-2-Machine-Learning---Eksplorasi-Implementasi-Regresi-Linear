#%%
# Import Regresi Linear Sederhana
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#%%
#membuat data penggunaan bahan bakar dan emisi CO2
df = pd.read_csv('FuelConsumptionCo2.csv')
df.head()


# %%
# analisis kolom 'ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'CO2EMISSIONS'
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'CO2EMISSIONS']]
cdf.head(9)

#%%
# visualisasi plot regresi linear sederhana 
plt.scatter(cdf['FUELCONSUMPTION_CITY'], cdf['CO2EMISSIONS'], color='blue')
plt.title('Regresi Linear Sederhana')
plt.xlabel('Konsumsi Bahan Bakar (City)')
plt.ylabel('Emisi CO2')
plt.show()

# %%
# visualisasi plot regresi linear sederhana antara 'ENGINESIZE' dan 'CO2EMISSIONS'
plt.scatter(cdf['ENGINESIZE'], cdf['CO2EMISSIONS'], color='blue')
plt.title('Regresi Linear Sederhana')
plt.xlabel('Ukuran Mesin')
plt.ylabel('Emisi CO2')
plt.show()

#%%
# membagi data menjadi training dan testing
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

#%%
# visualisasi data training antara 'ENGINESIZE' dan 'CO2EMISSIONS'
plt.scatter(train['ENGINESIZE'], train['CO2EMISSIONS'], color='blue')
plt.title('Data Training')
plt.xlabel('Ukuran Mesin')
plt.ylabel('Emisi CO2')
plt.show()

#%%
# membuat model regresi linear sederhana
regr = LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)

#%%
# koefisien regresi linear sederhana
print('Koefisien: ', regr.coef_)
print('Intercept: ', regr.intercept_)

#%%
# visualisasi plot regresi linear sederhana dengan data training
plt.scatter(train['ENGINESIZE'], train['CO2EMISSIONS'], color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.title('Regresi Linear Sederhana')
plt.xlabel('Ukuran Mesin')
plt.ylabel('Emisi CO2')
plt.show()
# %%

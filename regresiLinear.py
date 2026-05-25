#%%
# Regresi Linear Sederhana
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#%%
#membuat data penjualan dan harga
penjualan = np.array([6,5,5,4,4,3,2,2,2,1])
harga = np.array([16000,18000,27000,34000,50000,68000,65000,81000,85000,90000])


# %%
# print data penjualan dan harga
print("Data Penjualan: ", penjualan)
print("Data Harga: ", harga)

#%%
# visualisasi model regresi linear sederhana
plt.scatter(penjualan, harga)

#%%
# membuat model regresi linear sederhana
penjualan = penjualan.reshape(-1, 1)
linreg = LinearRegression()
linreg.fit(penjualan, harga)

#%%
# visualisasi plot regresi linear sederhana
plt.scatter(penjualan, harga, color='red')
plt.plot(penjualan, linreg.predict(penjualan), color='blue')
plt.title('Regresi Linear Sederhana')
plt.xlabel('Penjualan')
plt.ylabel('Harga')

# %%

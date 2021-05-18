import pandas as pd
import numpy as np
import os
import pip

GDPa = pd.read_csv("Real_GDP_ie_Annual.csv", index_col=0)
print(GDPa.head())
print(GDPa.tail())

FDI = pd.read_csv("FDI_Position_Ireland.csv", index_col=0)
print(FDI.head())
print(type(FDI))
print(FDI.shape)
print(FDI.dtypes)
FDI.dropna(subset=['NACE_Sector', 'Location', 'Mill_Eur'], inplace=True)

# Function FDI_All is FDI filtered by all NACE sectors and All countries
FDI_All = FDI.loc[(FDI['NACE_Sector'] == 'All NACE economic sectors') & (FDI['Location'] == 'All countries')]
print(FDI_All.head())
print(FDI_All.shape)

# GDP AND FDI_All together - merging by adding columns:
GDP_FDIall = pd.concat([GDPa, FDI_All], axis=1)
print(GDP_FDIall.head())

# Recreate table using np arrays and dropping NACE_Sector and Location:
GDP_Yr_l = GDP_FDIall['GDP_Yr'].tolist()
GDP_Yr_l = list(dict.fromkeys(GDP_Yr_l))
print(GDP_Yr_l)  # Yearly GDP list
GDP_YR_np = np.array(GDP_Yr_l)
print(GDP_YR_np)  # Yearly GDP numpy array

FDI_Yr_l_mill = GDP_FDIall['Mill_Eur'].tolist()
FDI_Yr_l_mill = list(dict.fromkeys(FDI_Yr_l_mill))
print(FDI_Yr_l_mill)  # Yearly FDI list
FDI_Yr_np_mill = np.array(FDI_Yr_l_mill)
print(FDI_Yr_np_mill)  # Yearly FDI numpy array

# Transform both columns to mill of Dollars by creating two functions:
GDPyrNP = GDP_YR_np / 1000000  # GDP per capita in mill of USD
FDIyrNP = FDI_Yr_np_mill * 1.21  # FDI per year in mill of USD by multiplying by FX

# Create a list for Years 2012 to 2019 with a while loop function:
Year = []
i = 2012
while len(Year) < 8:
    Year.append(i)
    i = i + 1
print(Year)

# Merge all three variables creating a dictionary and turning it into a Dataframe
Dict = {'GDP per Capita (mill USD)': GDPyrNP, 'FDI per Year (mill USD)': FDIyrNP}
GDPFDI = pd.DataFrame(Dict)
print(GDPFDI)
GDPFDI.index = Year
print(GDPFDI)  # Index the year

# Time Series Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots()
ax.plot(Year, GDPyrNP, color='blue')
ax.set_xlabel('year')
ax.set_ylabel('GDP per Capita (mill of Dollars)', color='blue')
ax2 = ax.twinx()
ax2.plot(Year, FDIyrNP, color='red')
ax2.set_ylabel('FDI per Year (mill of Dollars)', color='red')
plt.savefig("TimeSeries_GDP&FDI")
plt.show()

# Scatterplot:
sns.scatterplot(data=GDPFDI, x='FDI per Year (mill USD)', y='GDP per Capita (mill USD)')
plt.savefig("Scatterplot_GDP&FDI")
plt.show()

# Bubbleplot over time - size of bubbles = FDI
gdpfdi1 = pd.concat([GDPa, FDI], axis=1) # merge both initial datasets again, including all values in NACE_Sector and Location
gdpfdi = gdpfdi1.rename(columns = {'Mill_Eur': 'FDI'}, inplace=False) # rename column 'Mill Eur' to FDI
sns.scatterplot(data=gdpfdi, x='Year', y='GDP_Yr', size='FDI', alpha=0.5, sizes=(20, 800))
plt.xlabel("year")
plt.ylabel("GDP per Capita (mill of Dollars)")
plt.title("GDP per Capita (mill of Dollars) and size of FDI")
plt.savefig("Bubble Chart_GDP&FDI")
plt.show()

# All three plots show clear correlation between Foreign Direct Investment into Ireland and the size of GDP per Capita.
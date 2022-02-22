from ast import Index
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
data = pd.read_csv('owid-covid-data.csv')
AFG = data.loc[data['iso_code'] == "AFG"]
totalvaxx = AFG["total_vaccinations"]


iso_Codes = np.unique(data["iso_code"])
array = np.zeros((len(iso_Codes),2))

for i in range(len(iso_Codes)):
    array[i, 0] = data[data['iso_code'] == iso_Codes[i]]['people_fully_vaccinated_per_hundred'].max() 
    
    array[i, 1] = data[data['iso_code'] == iso_Codes[i]]['total_deaths_per_million'].max()

array = array[~np.isnan(array).any(axis=1)]

plt.scatter(array[:,0],array[:,1])
print(np.corrcoef(array[:,0], array[:,1]))
plt.xlabel("total_vaccinations_per_hundred")
plt.ylabel("total_deaths_per_million")
plt.show()

'''newArray = np.zeros((len(iso_Codes)))
for i in range(len(iso_Codes)):
    a = data[data['iso_code'] == iso_Codes[i]]['people_fully_vaccinated_per_hundred']
    b = data[data['iso_code'] == iso_Codes[i]]['new_deaths_smoothed_per_million']
    newArray[i] = a.corr(b)
newArray = newArray[~np.isnan(newArray)]
print(np.median(newArray))
plt.boxplot(newArray,notch = True)
plt.show()
'''
'''
def stringToNumber(string):
    power, output = len(string)-1, 0
    for i, char in enumerate(string):
        output += (ord(char)*(256**power))
        power -= 1

    return output

##not really sure why we need to make the function for part a 
##because dictionary keys can be strings in python, so it just
##makes me have to write a function to reverse the number encoding
##so thats great
def numberToString(number):
    q = 1
    string = ''
    while number // (256**q) > 0:
        q +=1
    q -=1
    while q >= 0:
        quo, number = divmod(number,256**q)
        string += chr(quo)
        q -= 1
    return string

totalDeathsDict = {}

for i, isoCode in enumerate(iso_Codes):
    totalDeathsDict[stringToNumber(isoCode)] = data[data['iso_code'] == iso_Codes[i]]['total_deaths_per_million'].max()

values = list(totalDeathsDict.values())
q1, q3  = np.nanquantile(values,(0.25,0.75))
bestCountries = []
worstCountries = []
for key,value in totalDeathsDict.items():
    if value <= q1:
        bestCountries.append(numberToString(key))
    elif value >= q3:
        worstCountries.append(numberToString(key))
print("The best countries are: ")
print(bestCountries)
print("The worst countries are")
print(worstCountries)


geographic = pd.read_csv('countries_codes_and_coordinates.csv')
def deleteQuote(string):
    a = string.strip()
    return a.strip('"')

editedLatitude = geographic['Latitude (average)'].apply(deleteQuote)
editedCode = geographic['Alpha-3 code'].apply(deleteQuote)

def lookupLatitude(code):
    try:
        idx = editedCode[editedCode == code].index[0]
        return float(editedLatitude[idx])
    except IndexError:
        return np.nan

bestCountryLatitudes = np.array([])
worstCountryLatitudes = np.array([])
for country in bestCountries:
    bestCountryLatitudes = np.append(bestCountryLatitudes,lookupLatitude(country))
for country in worstCountries:
    worstCountryLatitudes =np.append(worstCountryLatitudes,lookupLatitude(country))

bestCountryLatitudes = bestCountryLatitudes[~np.isnan(bestCountryLatitudes)]
worstCountryLatitudes = worstCountryLatitudes[~np.isnan(worstCountryLatitudes)]
print("The Best Countries' Latitude Are: ")
print(bestCountryLatitudes)
print("The Worst Countries' Latitude Are: ")
print(worstCountryLatitudes)


print(np.median(bestCountryLatitudes))
print(np.median(worstCountryLatitudes))
plt.boxplot([bestCountryLatitudes, worstCountryLatitudes], positions = [1,1.6], labels = ["Best Country Latitudes", "Worst Country Latitudes"], notch= True)
plt.show()'''

import requests

#define variables for the URL

distrito = "lisboa" # replace with your desired value
DICO = "1106"
concelho = "lisboa"

# set URL of API

original_url = 'https://api.ipma.pt/' 

# concatenate the URL for all data we want to retrieve

url = original_url + 'open-data/observation/climate/evapotranspiration/' + distrito + '/et0-' + DICO + '-' + concelho + '.csv'

# request the data from the API

resp = requests.get(url)

# create a cvs file with the contents from the API

with open('people/Vasco Florentino/data.csv', 'w') as f:
    f.write(resp.text)

# prints in command line all the values

print(resp.text)
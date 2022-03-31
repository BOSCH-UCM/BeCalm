#!/usr/bin/env python
# coding: utf-8

# # Patients casebase generator for the BeCalm project
# 

# In[14]:


from random import seed, randint, choice, gauss, choices, gammavariate, randrange
import math
import numpy as np

# Fijamos la semilla aleatoria
seed(10)

# Constantes relacionadas con los atributos y sus posiciones en el vector
AGE = 0
GENDER = 1
SMOKE = 2
PPM = 3
HEIGHT = 4
CO2 = 5
SPO2 = 6
MASK_PRESSURE = 7

MALE = "hombre"
FEMALE = "mujer"
GENDER_OPTIONS = [MALE, FEMALE]
FEMALE_PROBABILITY = 0.5
MALE_PROBABILITY = 1 - FEMALE_PROBABILITY
GENDER_PROBABILITIES = [MALE_PROBABILITY, FEMALE_PROBABILITY]

SMOKER = "fumador"
NON_SMOKER = "no fumador"
SMOKING_OPTIONS = [SMOKER, NON_SMOKER]
SMOKER_PROBABILITY = 0.2
NON_SMOKER_PROBABILITY = 1 - SMOKER_PROBABILITY
SMOKING_PROBABILITIES = [SMOKER_PROBABILITY, NON_SMOKER_PROBABILITY]

# Creamos el vector de casos y los vectores de FVC para los sujetos
cases = []
forced_vital_capacity = []

# Fijamos el número de casos
number_cases = 5000

# Funciones auxiliares
def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

### 1) Generation of patient's features

#### Seed values

# In[15]:


for i in range(number_cases):
  cases.append([randint(15, 65), choice(GENDER_OPTIONS), choices(SMOKING_OPTIONS, weights=SMOKING_PROBABILITIES)[0], int(gauss(75, 4))])


# #### Height
# 

# In[16]:


for case in cases:
  value = gauss(171, 5) if case[GENDER] == MALE else gauss(159.5,5)
  case.append(truncate(value, 2))


# #### CO2 inside mask

# In[17]:


for case in cases:
  case.append(int(gauss(450, 10)))


# #### Peripheral oxygen saturation

# In[18]:


for case in cases:
  x_normal = 0.0
  variance_smoking = 0.0
  if case[GENDER] == MALE:
    x_normal = gauss(97.1, 0.5)
    variance_smoking = 0.7 if case[SMOKE] == SMOKER else 0
  else:
    x_normal = gauss(96.6, 0.5)
    variance_smoking = 0.4 if case[SMOKE] == SMOKER else 0
  variance_age = case[AGE] / 65
  case.append(truncate(x_normal - variance_age - variance_smoking, 2))


# ### FVC and Mask Internal Pressure

# In[19]:


def get_forced_vital_capacity_base(gender, age):
  x_age = []
  y_fvc = []
  if gender == MALE:
    if age in range(15, 20):
      x_age = [16, 20]
      y_fvc = [4, 4.5]
    elif age in range(20, 39):
      x_age = [20, 39]
      y_fvc = [4.5, 5.2]
    elif age in range(39, 59):
      x_age = [39, 59]
      y_fvc = [5.2, 4.7]
    elif age in range(59, 66):
      x_age = [59, 66]
      y_fvc = [4.7, 4.3]
  else:
    if age in range(15, 40):
      x_age = [16, 40]
      y_fvc = [3.8, 3.8]
    elif age in range(40, 50):
      x_age = [40, 50]
      y_fvc = [3.8, 3.5]
    elif age in range(50, 66):
      x_age = [50, 66]
      y_fvc = [3.5, 2.8]
  polyn = np.poly1d(np.polyfit(x_age, y_fvc, 1))
  return polyn(age)


# In[20]:


for case in cases:
  f = 1 if cases[SMOKE] == SMOKER else 0
  forced_vital_capacity.append(truncate(get_forced_vital_capacity_base(case[GENDER], case[AGE]) * (1 + f * 0.1), 2))


# In[21]:


for idx, case in enumerate(cases):
  case.append(int((forced_vital_capacity[idx] / (4.5 * 1000) + 1) * 101000))


# In[22]:


print(cases[0])


# ### 2) Examples of generated patients
# 
# 

# In[23]:


txt_descriptor = "Paciente {id} - FVC: {FVC}. {gender} {smoke}, {age} años y {height} cm. \n\t En reposo: {ppm} pulsaciones por minuto, saturación periférica de oxígeno de {spO2}%, CO2 emitido de {CO2} ppm y presión en máscara de {pressure} Pa."
for idx, case in enumerate(cases):
  print(txt_descriptor.format(id=idx, FVC=forced_vital_capacity[idx], gender=case[GENDER], smoke=case[SMOKE], age=case[AGE], height=case[HEIGHT], ppm=case[PPM], spO2=case[SPO2], CO2=case[CO2], pressure=case[MASK_PRESSURE]) + "\n")


# ## 3) Time series generation

# In[24]:


print(cases[0])


# In[25]:


import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# Rango de fechas de los datos
date_range = pd.date_range(start='1/1/2020', end='1/2/2020', freq='H')

# Columnas del dataframe
DF_DATE = 'date'
DF_HEARTBEAT = 'pulsaciones por minuto'
DF_CO2 = 'CO2 emmitido'
DF_SPO2 = 'SpO2'
DF_PRESSURE = 'presión en máscara'
DF_DATETIME = 'datetime'

# Variaciones en reposo
rest_variability_ppm_up = 100
rest_variability_ppm_bottom = 60
rest_variability_ppm_mu = 0
rest_variability_ppm_sigma = 2

rest_variability_pressure_up = 101100
rest_variability_pressure_bottom = 100900
rest_variability_pressure_mu = 0
rest_variability_pressure_sigma = 5

rest_variability_co2_up = 900
rest_variability_co2_bottom = 300
rest_variability_co2_mu = 0
rest_variability_co2_sigma = 7

rest_variability_spo2_up = 100
rest_variability_spo2_bottom = 93
rest_variability_spo2_mu = 0
rest_variability_spo2_sigma = 0.75

# Variaciones en incremento/decremento
increment_variability_ppm_up = 130
increment_variability_ppm_bottom = 80
increment_variability_ppm_alpha = 3
increment_variability_ppm_beta = 1.5

increment_variability_pressure_up = 101200
increment_variability_pressure_bottom = 101000
increment_variability_pressure_alpha = 10
increment_variability_pressure_beta = 1.2

increment_variability_co2_up = 1000
increment_variability_co2_bottom = 600
increment_variability_co2_alpha = 25
increment_variability_co2_beta = 1.2



decrement_variability_ppm_up = 80
decrement_variability_ppm_bottom = 30
decrement_variability_ppm_alpha = 3
decrement_variability_ppm_beta = 1.5

decrement_variability_pressure_up = 101000
decrement_variability_pressure_bottom = 100800
decrement_variability_pressure_alpha = 25
decrement_variability_pressure_beta = 1.2

decrement_variability_co2_up = 400
decrement_variability_co2_bottom = 10
decrement_variability_co2_alpha = 25
decrement_variability_co2_beta = 1.2

decrement_variability_spo2_up = 90
decrement_variability_spo2_bottom = 85
decrement_variability_spo2_alpha = 1.8
decrement_variability_spo2_beta = 1.45





# Condiciones de estados futuros
from enum import Enum
class PatientState(Enum):
  REST = 0
  FAILURE_INCREMENT = 1
  FAILURE_DECREMENT = 2

# Funciones para generar futuros estados a partir de un estado de reposo
def generate_rest_sample(data_length, first_data, lower_bound, upper_bound, mu, sigma):
  serie = [first_data]
  for i in range(data_length):
    new_val = serie[i] + gauss(mu, sigma)
    if new_val > upper_bound:
      new_val = upper_bound + math.sin(i)*sigma*2#serie[i] - gauss(mu, sigma)
    if new_val < lower_bound:
      new_val = lower_bound + math.sin(i)*sigma*2#serie[i] + gauss(mu, sigma)
    serie.append(new_val)
  return serie

def generate_variate_serie_sample(data_length, state, first_data, lower_bound, upper_bound, alpha, beta, mu, sigma):
  serie = [first_data]
  operation = -1 if state == PatientState.FAILURE_DECREMENT else 1
  for i in range(data_length-1):
    new_val = serie[i] + operation * (gammavariate(alpha, beta)-alpha) + math.sin(i*2) * alpha *.5
    if (new_val > upper_bound and state == PatientState.FAILURE_INCREMENT):
        new_val = upper_bound + math.sin(i*2) * alpha *.5
    if (new_val < lower_bound and state == PatientState.FAILURE_DECREMENT):
        new_val = lower_bound - math.sin(i*2) * alpha *.5
    serie.append(new_val)
  limit_rest_serie = generate_rest_sample(data_length - len(serie), serie[len(serie)-1], lower_bound, upper_bound, mu, sigma)
  return serie + limit_rest_serie


# In[26]:


def prepare_df_times():
  patient_df = pd.DataFrame(date_range, columns=[DF_DATE])
  patient_df[DF_DATETIME] = pd.to_datetime(patient_df[DF_DATE])
  patient_df = patient_df.set_index(DF_DATETIME)
  patient_df.drop([DF_DATE], axis=1, inplace=True)
  return patient_df

def information_text_generator(variability_parameter, parameter_text):
  if variability_parameter == PatientState.FAILURE_DECREMENT:
      return "niveles de " + parameter_text + " aumentan"
  elif variability_parameter == PatientState.FAILURE_INCREMENT:
      return "niveles de " + parameter_text + " disminuyen"
  else:
      return "niveles de " + parameter_text + " en reposo"

def plot_patient(patient_vector, idx, co2_variability=PatientState.REST, spO2_variability=PatientState.REST, pressure_variability=PatientState.REST, heartbeat_variability=PatientState.REST):
  heartbeat_df = prepare_df_times()
  co2_df = prepare_df_times()
  spo2_df = prepare_df_times()
  pressure_df = prepare_df_times()

  heartbeat = patient_vector[PPM]
  co2 = patient_vector[CO2]
  spO2 = patient_vector[SPO2]
  pressure = patient_vector[MASK_PRESSURE]
  
  
  generate_variate_serie_sample(len(date_range), PatientState.FAILURE_INCREMENT, heartbeat, increment_variability_ppm_bottom, increment_variability_ppm_up, increment_variability_ppm_alpha, increment_variability_ppm_beta, rest_variability_ppm_mu, rest_variability_ppm_sigma)
  


# In[ ]:





# ## Generación del fichero de series temporales

# In[27]:


import sys
heartbeat_decrement_list = []
co2_decrement_list = []
spo2_decrement_list = []
pressure_decrement_list = []

for idx, patient_vector in enumerate(cases):
    heartbeat = patient_vector[PPM]
    co2 = patient_vector[CO2]
    spo2 = patient_vector[SPO2]
    pressure = patient_vector[MASK_PRESSURE]
    
    heartbeat_serie = generate_variate_serie_sample(len(date_range), PatientState.FAILURE_DECREMENT, heartbeat, decrement_variability_ppm_bottom, decrement_variability_ppm_up, decrement_variability_ppm_alpha, decrement_variability_ppm_beta, rest_variability_ppm_mu, rest_variability_ppm_sigma)
    heartbeat_decrement_list.append(heartbeat_serie)
    
    co2_serie = generate_variate_serie_sample(len(date_range), PatientState.FAILURE_DECREMENT, co2, decrement_variability_co2_bottom, decrement_variability_co2_up, decrement_variability_co2_alpha, decrement_variability_co2_beta, rest_variability_co2_mu, rest_variability_co2_sigma)
    co2_decrement_list.append(co2_serie)

    pressure_serie = generate_variate_serie_sample(len(date_range), PatientState.FAILURE_DECREMENT, pressure, decrement_variability_pressure_bottom, decrement_variability_pressure_up, decrement_variability_pressure_alpha, decrement_variability_pressure_beta, rest_variability_pressure_mu, rest_variability_pressure_sigma)
    pressure_decrement_list.append(pressure_serie)

    spo2_serie = generate_variate_serie_sample(len(date_range), PatientState.FAILURE_DECREMENT, spo2, decrement_variability_spo2_bottom, decrement_variability_spo2_up, decrement_variability_spo2_alpha, decrement_variability_spo2_beta, rest_variability_spo2_mu, rest_variability_spo2_sigma)
    spo2_decrement_list.append(spo2_serie)


# ### Series temporales en incremento

# In[28]:


heartbeat_increment_list = []
co2_increment_list = []
spo2_increment_list = []
pressure_increment_list = []

for idx, patient_vector in enumerate(cases):
    heartbeat = patient_vector[PPM]
    co2 = patient_vector[CO2]
    spO2 = patient_vector[SPO2]
    pressure = patient_vector[MASK_PRESSURE]

    
    heartbeat_serie = generate_variate_serie_sample(len(date_range), PatientState.FAILURE_INCREMENT, heartbeat, increment_variability_ppm_bottom, increment_variability_ppm_up, increment_variability_ppm_alpha, increment_variability_ppm_beta, rest_variability_ppm_mu, rest_variability_ppm_sigma)
    heartbeat_increment_list.append(heartbeat_serie)
    
    co2_serie = generate_variate_serie_sample(len(date_range), PatientState.FAILURE_INCREMENT, co2, increment_variability_co2_bottom, increment_variability_co2_up, increment_variability_co2_alpha, increment_variability_co2_beta, rest_variability_co2_mu, rest_variability_co2_sigma)
    co2_increment_list.append(co2_serie)

    pressure_serie = generate_variate_serie_sample(len(date_range), PatientState.FAILURE_INCREMENT, pressure, increment_variability_pressure_bottom, increment_variability_pressure_up, increment_variability_pressure_alpha, increment_variability_pressure_beta, rest_variability_pressure_mu, rest_variability_pressure_sigma)
    pressure_increment_list.append(pressure_serie)


# ### Series temporales en reposo

# In[29]:


heartbeat_rest_list = []
co2_rest_list = []
spo2_rest_list = []
pressure_rest_list = []

for idx, patient_vector in enumerate(cases):
    heartbeat = patient_vector[PPM]
    co2 = patient_vector[CO2]
    spo2 = patient_vector[SPO2]
    pressure = patient_vector[MASK_PRESSURE]
    
    heartbeat_serie = generate_rest_sample(len(date_range), heartbeat, rest_variability_ppm_bottom, rest_variability_ppm_up, rest_variability_ppm_mu, rest_variability_ppm_sigma)
    heartbeat_rest_list.append(heartbeat_serie)
    
    co2_serie = generate_rest_sample(len(date_range), co2, rest_variability_co2_bottom, rest_variability_co2_up, rest_variability_co2_mu, rest_variability_co2_sigma)
    co2_rest_list.append(co2_serie)

    spo2_serie = generate_rest_sample(len(date_range), spo2, rest_variability_spo2_bottom, rest_variability_spo2_up, rest_variability_spo2_mu, rest_variability_spo2_sigma)
    spo2_rest_list.append(spo2_serie)

    pressure_serie = generate_rest_sample(len(date_range), pressure, rest_variability_pressure_bottom, rest_variability_pressure_up, rest_variability_pressure_mu, rest_variability_pressure_sigma)
    pressure_rest_list.append(pressure_serie)


# ## Save files

# In[30]:


save = False
if save:
    list_cases = cases.copy()
    for idx, case in enumerate(list_cases):
      case.insert(0, idx)
    np.savetxt("patients_data.csv", list_cases, delimiter =", ", fmt ='% s')

    np.savetxt("heartbeat_decrement.csv", heartbeat_decrement_list, delimiter =", ", fmt ='% s')
    np.savetxt("co2_decrement.csv", co2_decrement_list, delimiter =", ", fmt ='% s')
    np.savetxt("pressure_decrement.csv", pressure_decrement_list, delimiter =", ", fmt ='% s')
    np.savetxt("spo2_decrement.csv", spo2_decrement_list, delimiter =", ", fmt ='% s')

    np.savetxt("heartbeat_increment.csv", heartbeat_increment_list, delimiter =", ", fmt ='% s')
    np.savetxt("co2_increment.csv", co2_increment_list, delimiter =", ", fmt ='% s')
    np.savetxt("pressure_increment.csv", pressure_increment_list, delimiter =", ", fmt ='% s')

    np.savetxt("spo2_rest.csv", spo2_rest_list, delimiter =", ", fmt ='% s')
    np.savetxt("pressure_rest.csv", pressure_rest_list, delimiter =", ", fmt ='% s')
    np.savetxt("co2_rest.csv", co2_rest_list, delimiter =", ", fmt ='% s')
    np.savetxt("heartbeat_rest.csv", heartbeat_rest_list, delimiter =", ", fmt ='% s')


# ## Plot cases

# In[2]:


import pylab as pl

def plotSeries(series, numLines, title):
    pl.figure(figsize=(14,6))
    for s in series[0][:numLines]:
        pl.plot(s, 'r')
    for s in series[1][:numLines]:
        pl.plot(s, 'g', alpha=.5)
    for s in series[2][:numLines]:
        pl.plot(s, 'b', alpha=.5)
    pl.title(title)
    pl.show()


# In[32]:


plotSeries([heartbeat_rest_list, heartbeat_increment_list, heartbeat_decrement_list ], 100, 'RHR')


# In[33]:


plotSeries([pressure_rest_list, pressure_increment_list, pressure_decrement_list ], 100, 'MIP')


# In[34]:


plotSeries([co2_rest_list, co2_increment_list, co2_decrement_list ], 100, 'Co2')


# In[40]:


pl.figure(figsize=(14,6))
for s in spo2_rest_list[:100]:
  pl.plot(s, 'r')
for s in spo2_decrement_list[:100]:
  pl.plot(s, 'b')

pl.title('MIP')


pl.show()


# In[36]:


import random 
import matplotlib.pyplot as plt 
def plotGamma(subplt, alpha, beta, _title =""):
  nums = [] 
  for i in range(10000): 
    temp = random.gammavariate(alpha, beta)-alpha
    nums.append(temp) 
  subplt.hist(nums, bins = 200) 
  subplt.set_title(_title + " alpha:"+str(alpha) + ", beta:"+ str(beta))
    
def plotGauss(subplt, mu, sigma, _title =""):
  nums = [] 
  for i in range(10000): 
    temp = gauss(mu, sigma)
    nums.append(temp) 
  subplt.hist(nums, bins = 200) 
  subplt.set_title(_title + " mu:"+str(mu) + ", sigma:"+ str(sigma))


# In[37]:


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
#plt.figure(figsize=(4,3), tight_layout=True, dpi=300)       
plotGamma(ax1, increment_variability_ppm_alpha,increment_variability_ppm_beta, "RHR")
plotGamma(ax2, increment_variability_pressure_alpha,increment_variability_pressure_beta, "MIP")
plotGamma(ax3, increment_variability_co2_alpha,increment_variability_co2_beta, "CO2")
plotGamma(ax4, decrement_variability_spo2_alpha,decrement_variability_spo2_beta, "SpO2")
fig.set(dpi=300,tight_layout=True, size_inches=(6,6) )
fig.savefig('gammas.png')
plt.show()
#plotGamma(25,1.2, "co2 alpha:25, beta:1.2")
#plotGamma(25,1.2)


# In[38]:


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
#plt.figure(figsize=(4,3), tight_layout=True, dpi=300)       
plotGauss(ax1, rest_variability_ppm_mu,rest_variability_ppm_sigma, "RHR")
plotGauss(ax2, rest_variability_pressure_mu,rest_variability_pressure_sigma, "MIP")
plotGauss(ax3, rest_variability_co2_mu,rest_variability_co2_sigma, "CO2")
plotGauss(ax4, rest_variability_spo2_mu,rest_variability_spo2_sigma, "SpO2")
fig.set(dpi=300,tight_layout=True, size_inches=(6,6) )
fig.savefig('gauss.png')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





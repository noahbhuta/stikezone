import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

pitch_data = pd.read_csv('2019_pitches.csv')

fig, ax = plt.subplots()

print(pitch_data.columns)
print(pitch_data.type.value_counts())

def strikeout(pitches):
  pitches.type = pitches.type.map({'B':0,'S':1,'*B':0,'C':1,'W':1,'M':1,'H':0})
  print(pitches.type)

  print(pitches['px'])
  print(pitches['pz'])

  pitches = pitches.dropna(subset = ["type",'px','pz'])

  plt.scatter(pitches.px, pitches.pz, c = pitches.type, cmap = plt.cm.coolwarm,alpha = .25)


  training_set ,validation_set = train_test_split(pitches,random_state  = 1)
  classifier = SVC(gamma = 100,C = 100)
  classifier.fit(training_set[['px','pz']],training_set.type)
  draw_boundary(ax,classifier)
  plt.show()
  print(classifier.score(validation_set[['px','pz']],validation_set.type))

  gamma_list = []
  for gam in range(1,100):
    classifier = SVC(gamma = gam,C = 100)
    classifier.fit(training_set[['px','pz']],training_set.type)
    gamma_list.append(classifier.score(validation_set[['px','pz']],validation_set.type))
  maximum = max(gamma_list)
  gamma_value = gamma_list.index(maximum)+1
  print(maximum,gamma_value)


  C_list = []
  for c in range(1,100):
    classifier = SVC(gamma = gamma_value,C = c)
    classifier.fit(training_set[['px','pz']],training_set.type)
    C_list.append(classifier.score(validation_set[['px','pz']],validation_set.type))
  maximum = np.max(C_list)
  C_value = C_list.index(maximum)+1
  print(maximum,C_value)
strikeout(pitch_data)

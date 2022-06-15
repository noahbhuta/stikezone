import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from svm_visualization import draw_boundary
from players import aaron_judge, jose_altuve, david_ortiz

fig, ax = plt.subplots()
print(aaron_judge.columns)
print(aaron_judge.description.unique())
print(aaron_judge.type.unique())


def player_strikeout(player):
  player.type = player.type.map({'B':0,'S':1})
  print(player.type)

  print(player['plate_x'])
  print(player['plate_z'])

  player = player.dropna(subset = ["type",'plate_x','plate_z'])

  plt.scatter(player.plate_x, player.plate_z, c = player.type, cmap = plt.cm.coolwarm,alpha = .25)


  training_set ,validation_set = train_test_split(player,random_state  = 1)
  classifier = SVC(gamma = 100,C = 100)
  classifier.fit(training_set[['plate_x','plate_z']],training_set.type)
  draw_boundary(ax,classifier)
  plt.show()
  print(classifier.score(validation_set[['plate_x','plate_z']],validation_set.type))

  gamma_list = []
  for gam in range(1,100):
    classifier = SVC(gamma = gam,C = 100)
    classifier.fit(training_set[['plate_x','plate_z']],training_set.type)
    gamma_list.append(classifier.score(validation_set[['plate_x','plate_z']],validation_set.type))
  maximum = max(gamma_list)
  gamma_value = gamma_list.index(maximum)+1
  print(maximum,gamma_value)


  C_list = []
  for c in range(1,100):
    classifier = SVC(gamma = gamma_value,C = c)
    classifier.fit(training_set[['plate_x','plate_z']],training_set.type)
    C_list.append(classifier.score(validation_set[['plate_x','plate_z']],validation_set.type))
  maximum = max(C_list)
  C_value = C_list.index(maximum)+1
  print(maximum,C_value)
player_strikeout(aaron_judge)

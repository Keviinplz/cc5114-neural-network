# Importaciones Nativas
import re
import random

# Importaciones de Librerias
import numpy as np


class Dataset():
  """
  Define a Dataset structure
  """

  def __init__(self, filepath, regex, group):
    self.filepath = filepath
    self.regex = regex
    self.group = group

  def extractInfoFromFile(self, filepath: str, regex: str, group: list) -> list:
    """
    Extract info from File use Regular Expresion
    """
    inputsList = []
    outputsList = []
    lines = open(filepath).read().splitlines()
    for i in range(1, len(lines)):
      match = re.search(regex, lines[i].rstrip('\n'), re.MULTILINE) # Search line by line
      inputData = []
      for i in range(len(group) - 1):
        x = float(match.group(group[i])[:-1])
        inputData.append(x)
      inputsList.append(inputData)
      y = float(match.group(len(group) + 1))
      outputsList.append(np.array([y]))

    data = []
    for x, y in zip(inputsList, outputsList):
      data.append([np.array(x), np.array(y)])
    return data # The data has the form: [input, output]

  def getData(self):
    data = self.extractInfoFromFile(self.filepath, self.regex, self.group)
    #random.shuffle(data)
    X = []
    Y = []
    for x, y in data:
      X.append(x)
      Y.append(y)

    return np.array(X), np.array(Y)



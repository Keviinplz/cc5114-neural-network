# Importaciones Nativas
import re
import random

# Importaciones de Librerias
import numpy as np


class Dataset():
  """
  Define a Dataset structure
  """

  def __init__(self, filepath):
    self.filepath = filepath
    self.regex = r'^(\w[#b]|\w);(.+);".+";((\w.){5}\w);.+$'
    self.diccChords = {}

  @staticmethod
  def encodeX(n):
    return -1 if n == 'x' else int(n)

  def getChords(self) -> dict:
    return self.diccChords

  def updateDiccChords(self, char: str) -> list:
    if char not in self.diccChords.keys():
      self.diccChords[char] = len(self.diccChords.keys())

  def oneHotEncoding(self, char: str) -> list:
    output = np.zeros(len(self.diccChords.keys()))
    position = self.diccChords[char]
    output[position] = 1
    return output

  def extractInfoFromFile(self, filepath: str, regex: str) -> list:
    """
    Extract info from File use Regular Expresion
    """
    inputsList = []
    outputsList = []
    lines = open(filepath).read().splitlines()
    for i in range(1, len(lines)):
      match = re.search(regex, lines[i].rstrip('\n'), re.MULTILINE) # Search line by line
      inputData = list(map(self.encodeX, match.group(3).split(','))) # transform input to list, with x = -1 (encode)
      self.updateDiccChords(match.group(1))
      inputsList.append(inputData)
      outputsList.append(match.group(1))
    
    outputsList = list(map(self.oneHotEncoding, outputsList))

    data = []
    for x, y in zip(inputsList, outputsList):
      data.append([np.array(x),y])
    return data # The data has the form: [input, output]

  def getData(self):
    data = self.extractInfoFromFile(self.filepath, self.regex)
    random.shuffle(data)
    X = []
    Y = []
    for x, y in data:
      X.append(x)
      Y.append(y)

    return np.array(X), np.array(Y)



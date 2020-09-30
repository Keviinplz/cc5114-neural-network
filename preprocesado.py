import re

# Constantes
FILE_PATH = 'dataset/data.csv'
REGULAR_EXPRESION = r'^(\w[#b]|\w);(.+);".+";((\w.){5}\w);.+$'

# Abrimos el dataset y lo guardamos en un string
with open(FILE_PATH, 'r') as file:
    file_to_string = file.read()

# Usando expresiones regulares, extraemos la información relevante del dataset
dirty_data = re.findall(REGULAR_EXPRESION, file_to_string, re.MULTILINE)

# Limpiamos los datos para darle un formato de tipo
# dato = [input, output]

clean_data = []
for data in dirty_data:
    cosa = [data[2], data[0] + data[1]]
    clean_data.append(cosa)

# Podemos usar 80% Training y 20% Test

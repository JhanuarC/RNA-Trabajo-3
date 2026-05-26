import kagglehub
import pandas as pd
import os

# Descargar automáticamente
path = kagglehub.dataset_download("amanmehra23/travel-recommendation-dataset")

print(path)

# Ver archivos disponibles
print(os.listdir(path))


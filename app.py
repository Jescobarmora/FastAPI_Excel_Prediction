from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from pycaret.classification import predict_model
import json
import tempfile
import pandas as pd
import uvicorn
import pickle
import shutil
import traceback

path = "/code/Python/Corte_2/Quiz_2_2/Punto_3/"

# Crear una instancia de FastAPI
app = FastAPI()

# Definir el archivo JSON donde se guardarán las predicciones
file_name = 'predictions/predictions.json'

with open(path + 'models/ridge_model.pkl', 'rb') as file:
  modelo = pickle.load(file)


class InputData(BaseModel):
  Email: str
  Address: str
  Dominio: str
  Tecnologia: str
  Avg_Session_Length: float
  Time_on_App: float
  Time_on_Website: float
  Length_of_Membership: float


@app.get("/")
def home():
  # Retorna un simple mensaje de texto
  return 'Predicción estudiantes'


# Función para guardar predicciones en un archivo JSON
def save_prediction(prediction_data):
  try:
    with open(path + file_name, 'r') as file:
      predictions = json.load(file)
  except (FileNotFoundError, json.JSONDecodeError):
    predictions = []

  predictions.append(prediction_data)

  with open(path + file_name, 'w') as file:
    json.dump(predictions, file, indent=4)

# Definir un endpoint para manejar la subida de archivos Excel y hacer predicciones
@app.post("/upload-excel")
def upload_excel(file: UploadFile = File(...)):
    try:
        # Crear un archivo temporal para manejar el archivo subido
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            shutil.copyfileobj(file.file, temp_file)

            # Leer el archivo Excel usando pandas y almacenarlo en un DataFrame
            df = pd.read_excel(temp_file.name)

            # Limpiar y convertir las columnas deseadas (con comas en lugar de puntos)
            def clean_and_convert(column_name):
                df[column_name] = pd.to_numeric(df[column_name].astype(str).str.replace(',', '.'), errors='coerce')

            # Limpiar y convertir las columnas numéricas deseadas
            for column in ['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']:
                clean_and_convert(column)

            # predictions = predict_model(modelo, data=df)
            # predictions['price'] = predictions["prediction_label"]
            # prediction_prices_list = list(predictions['price'])
            
            predictions = predict_model(modelo, data=df)
            # Redondear a 4 decimales las predicciones en la columna "prediction_label"
            predictions['price'] = predictions["prediction_label"].round(4)
            prediction_prices_list = list(predictions['price'])


            return {"predictions": prediction_prices_list}

    except Exception as e:
      return {"error": f"Ocurrió un error: {str(e)}", "traceback": traceback.format_exc()}
      
# Ejecutar la aplicación si se llama desde la terminal
if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8000)
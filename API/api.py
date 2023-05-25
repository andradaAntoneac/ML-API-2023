import os

import uvicorn as uvicorn
from fastapi import FastAPI
from fastapi import UploadFile
import api_preprocess
import tensorflow as tf

model_path = "model"
model = tf.keras.models.load_model(model_path)

app = FastAPI()


@app.post("/uploadfile/{patient_id}/{file_id}/file")
def create_upload_file(patient_id: int, file_id: int, file: UploadFile):
    contents = file.file.read()
    if not os.path.exists('patient_data'):
        os.makedirs('patient_data')
    wav_path = f'patient_data/{patient_id}_{file_id}.wav'
    f = open(wav_path, 'wb')
    f.write(contents)
    f.close()
    wav_proc = api_preprocess.WAVproc()
    wav_proc.filtering(wav_path)
    predicted = wav_proc.get_verdict()
    wav_proc.cleanup()
    if not predicted:
        return wav_proc.not_predicted_json_compute()
    return wav_proc.predicted_json_compute()


if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=9000, log_level="info")

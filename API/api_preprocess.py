import time
from enum import Enum
import base64
import librosa
import numpy
import numpy as np
from scipy.signal import butter, lfilter
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import os

from api import model


class Channel(Enum):
    HEALTHY = 0
    HYPOPNEA = 1
    MIXED_APNEA = 2
    OBSTRUCTIVE_APNEA = 3


class WAVproc:
    def __init__(self):
        self.no_of_samples = None
        self.events = []
        self.final_signal = None
        self.wav_path = None
        self.sampling_rate = None
        self.counters = [0, 0, 0, 0]
        self.prediction = None
        self.waveforms_path = []

    def filtering(self, path):
        t1 = time.perf_counter()
        self.wav_path = path
        sr = 6300
        y, s = librosa.load(path, sr=sr)
        self.sampling_rate = librosa.get_samplerate(path)

        # Sample rate and desired cutoff frequencies (in Hz).
        fs = 6300.0
        lowcut = 100.0
        highcut = 3000.0
        filtered_y = self.butter_bandpass_filter(y, lowcut, highcut, fs, order=4)

        # Resample + low pass filter
        y_5k = librosa.resample(filtered_y, orig_sr=sr, target_sr=5000)
        cutoff = 20
        freqRatio = cutoff / self.sampling_rate

        N = int(math.sqrt(0.196201 + freqRatio ** 2) / freqRatio)

        cumsum = numpy.cumsum(numpy.insert(y_5k, 0, 0))
        self.final_signal = ((cumsum[N:] - cumsum[:-N]) / N)

        t2 = time.perf_counter()
        print(f"Filtering was made in {t2 - t1} seconds")

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        return butter(order, [lowcut, highcut], fs=fs, btype='band')

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def get_verdict(self):
        t1 = time.perf_counter()
        duration = librosa.get_duration(path=self.wav_path)
        self.no_of_samples = int(duration / 60)
        for i in range(0, self.no_of_samples + 1):
            current_signal = self.final_signal[(self.no_of_samples + i) * 60:(10 * 5000)]
            plt.figure()
            librosa.display.waveshow(current_signal, sr=5000)
            image_path = f'patient_data\\waveform{i}_second{i * 60}.jpg'
            plt.savefig(image_path)
            self.waveforms_path.append(image_path)
            image = tf.keras.preprocessing.image.load_img(image_path, target_size=(640, 480))
            input_arr = tf.keras.preprocessing.image.img_to_array(image)
            input_arr = np.array([input_arr])
            response = model.predict(input_arr)
            response_arr = response[0].tolist()
            max_val = max(response_arr)
            if max_val > 0.5:
                index_max = response_arr.index(max_val)
                ind = Channel(index_max)
                with open(image_path, "rb") as img_file:
                    image64 = base64.b64encode(img_file.read())
                self.events.append({"event": ind.name, "second": f'{i * 60}', "image64": image64})
                for channel_type in [channel.value for channel in Channel]:
                    if ind.value == channel_type:
                        self.counters[channel_type] += 1
        self.counters = [counter / len(self.events) for counter in self.counters]
        max_val = max(self.counters)
        if max_val < 0.75:
            predicted = False
        else:
            predicted = True
            self.prediction = Channel(self.counters.index(max_val))
        t2 = time.perf_counter()
        print(f"Getting the verdict was made in {t2 - t1} seconds")
        return predicted

    def not_predicted_json_compute(self):
        return {"predicted": "false", "events": self.events}

    def predicted_json_compute(self):
        return {"predicted": "true", "verdict": self.prediction.name}

    def cleanup(self):
        for file in self.waveforms_path:
            os.remove(file)
        os.remove(self.wav_path)

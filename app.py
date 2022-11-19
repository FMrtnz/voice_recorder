#import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

# App import
from flask import Flask,render_template,url_for,send_from_directory,redirect,request
# Save the sound
import sounddevice as sd
from scipy.io.wavfile import write

# Imports used through the rest of the notebook.
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

#from tortoise.api import TextToSpeech
#from tortoise.utils.audio import load_audio, load_voice, load_voices

# This will download all the models used by Tortoise from the HuggingFace hub.
#tts = TextToSpeech()

app = Flask(__name__)

file_path = './static/audio/'
file_name = 'output.wav'
file_name_2 = 'output2.wav'

@app.route('/')

def main():
    files = [file_name, file_name_2]
    return render_template('index.html', audios=files)

@app.route('/record_audio/<file_name>')

def record_audio(file_name):
    fs = 44100  # Sample rate
    seconds = 10  # Duration of recording
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    write(file_path + file_name, fs, myrecording)  # Save as WAV file
    return redirect(url_for('main'))

@app.route('/generate_audio', methods=['POST'])

def generate_audio():
    text = request.form['text_to_speech'] if request.method == 'POST' else ""
    #voice_samples, conditioning_latents = load_voice('output', extra_voice_dirs=['static/audio'])
    #gen = tts.tts_with_preset(text, voice_samples=voice_samples,
    #conditioning_latents=conditioning_latents, preset='ultra_fast')
    #gen = tts.tts_with_preset(text, voice_samples=None, preset='fast')
    #torchaudio.save('generated.wav', gen.squeeze(0).cpu(), 24000)
    return text
    return redirect(url_for('main'))



app.run(debug = True) # to allow for debugging and auto-reload

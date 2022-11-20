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
torchaudio.set_audio_backend("soundfile")
import torch.nn as nn
import torch.nn.functional as F

# Import the package to transform the speech with voice
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices

# SAve to timestamp
import time

# This will download all the models used by Tortoise from the HuggingFace hub.
tts = TextToSpeech()

app = Flask(__name__)

# Define the main variables
file_path = './static/audio/'
file_name = 'output.wav'
file_name_2 = 'output2.wav'

# Display the home page
@app.route('/')
def main():
    files = [file_name, file_name_2]
    return render_template('index.html', audios=files)

# Save the record with the file
@app.route('/record_audio/<file_name>')
def record_audio(file_name):
    fs = 44100  # Sample rate
    seconds = 10  # Duration of recording
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    write(file_path + "/sample/" + file_name, fs, myrecording)  # Save as WAV file
    return redirect(url_for('main'))

# Display the result template
@app.route("/results")
def results():
    files = [file_name, file_name_2]
    final_file = request.args.get('final_file') if request.method == 'GET' else ""
    return render_template("results.html", audios=files, final_file=final_file)

@app.route('/generate_audio', methods=['POST'])
def generate_audio():
    # Set the text for speech
    text = request.form['text_to_speech'] if request.method == 'POST' else ""
    # Set the voice
    voice_samples, conditioning_latents = load_voice('sample', extra_voice_dirs=['static/audio'])
    # Generate the speech
    gen = tts.tts_with_preset(
        text # Define the text
        ,voice_samples=voice_samples # Set the voice
        ,conditioning_latents=conditioning_latents # Define voice
        ,preset='ultra_fast' # Set the speech speed
        )
    time_stamp = int(time.time())
    # Save the results with torch
    result_name = f"{time_stamp}-generated.wav"
    torchaudio.save(file_path + result_name, gen.squeeze(0).cpu(), 24000)
    return redirect(url_for( 'results', final_file=result_name ))

app.run(debug = True, port=5000) # to allow for debugging and auto-reload

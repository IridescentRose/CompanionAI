import io
from pydub import AudioSegment
import speech_recognition as sr
import whisper
import queue
import tempfile
import os
import threading
import click
import torch
import numpy as np
import pyttsx3
import requests
from pydub.playback import play
import openai
import json
from multiprocessing.connection import Listener

@click.command()
@click.option("--model", default="base", help="Model to use", type=click.Choice(["tiny","base", "small","medium","large"]))
@click.option("--energy", default=300, help="Energy level for mic to detect", type=int)
@click.option("--dynamic_energy", default=False,is_flag=True, help="Flag to enable dynamic engergy", type=bool)
@click.option("--pause", default=0.8, help="Pause time before entry ends", type=float)
def main(model, energy, pause,dynamic_energy):
    global EL
    global EL_key
    global EL_voice
    global OAI
    global OAI_key
    global Use_EL
    global AI_Name

    try:
        with open("config.json", "r") as json_file:
            data = json.load(json_file)
    except:
        print("Unable to open JSON file.")
        exit()

    class EL:
        key = data["keys"]["EL_KEY"]
        voice = data["data"]["EL_Voice"]

    Use_EL = data["data"]["Use_EL"]
    AI_Name = data["data"]["AI_Name"]

    class OAI:
        key = data["keys"]["OAI_KEY"]
        model = data["data"]["OAI_Model"]
        prompt = data["data"]["OAI_Prompt"]
        temperature = 0.9
        max_tokens = 100
        top_p = 1
        frequency_penalty = 1
        presence_penalty = 1

    #there are no english models for large
    if model != "large":
        model = model + ".en"

    audio_model = whisper.load_model(model)
    audio_queue = queue.Queue()
    result_queue = queue.Queue()

    threading.Thread(target=MicrophoneThread,
                     args=(audio_queue, energy, pause, dynamic_energy)).start()
    threading.Thread(target=ResponseLoop,
                     args=(audio_queue, result_queue, audio_model)).start()
    threading.Thread(target=InputThread, args=(result_queue,)).start()

    while True:
        print(result_queue.get())

def InputThread(result_queue):
    while True: 
        predicted_text = input("").lstrip().rstrip()

        if(predicted_text == "And now please quit."):
            os._exit(0)
        LLM_Submit(predicted_text, result_queue)

def MicrophoneThread(audio_queue, energy, pause, dynamic_energy):
    #load the speech recognizer and set the initial energy threshold and pause threshold
    r = sr.Recognizer()
    r.energy_threshold = energy
    r.pause_threshold = pause
    r.dynamic_energy_threshold = dynamic_energy

    with sr.Microphone(sample_rate=16000) as source:
        print("Say Something:")
        i = 0
        while True:
            #get and save audio to wav file
            audio = r.listen(source)
            
            torch_audio = torch.from_numpy(np.frombuffer(audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0)
            audio_data = torch_audio

            audio_queue.put_nowait(audio_data)
            i += 1

def LLM(message):
    openai.api_key = OAI.key
    response = openai.Completion.create(
      model= OAI.model,
      prompt= OAI.prompt + "\n\n#########\n" + message + "\n#########\n",
      temperature = OAI.temperature,
      max_tokens = OAI.max_tokens,
      top_p = OAI.top_p,
      frequency_penalty = OAI.frequency_penalty,
      presence_penalty = OAI.presence_penalty
    )

    json_object = json.loads(str(response))
    return(json_object['choices'][0]['text'])

def Setup_TTS():
    global engine
    engine = pyttsx3.init()
    engine.setProperty('rate', 180)
    engine.setProperty('volume', 1)
    voice = engine.getProperty('voices')
    engine.setProperty('voice', voice[1].id)
    print(engine)
    print(voice)

def REG_TTS(message):
    if engine._inLoop:
        engine.endLoop()
    
    engine.say(message)
    engine.runAndWait()


def EL_TTS(message):
    url = f'https://api.elevenlabs.io/v1/text-to-speech/{EL.voice}'
    headers = {
        'accept': 'audio/mpeg',
        'xi-api-key': EL.key,
        'Content-Type': 'application/json'
    }
    data = {
        'text': message,
        'voice_settings': {
            'stability': 0.75,
            'similarity_boost': 0.75
        }
    }

    response = requests.post(url, headers=headers, json=data, stream=True)
    audio_content = AudioSegment.from_file(io.BytesIO(response.content), format="mp3")
    play(audio_content)

def TTS(message):
    if(Use_EL):
        EL_TTS(message)
    else:
        REG_TTS(message)

def LLM_Submit(message, result_queue):
    result_queue.put_nowait(f'<You>: {message}')
    response = LLM(message)
    result_queue.put_nowait(f'<{AI_Name}>: {response.lstrip()}')
    #TTS(response)

def ResponseLoop(audio_queue, result_queue, audio_model):
    Setup_TTS()

    while True:
        audio_data = audio_queue.get()
        result = audio_model.transcribe(audio_data,language='english')

        predicted_text = result["text"].lstrip()

        if(predicted_text == "And now please quit."):
            os._exit(0)
        
        LLM_Submit(predicted_text, result_queue)

main()
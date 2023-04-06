import openai
import json
import pyttsx3
import os
import threading
import torch
import whisper
import speech_recognition
import numpy as np
import io
import queue

from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain.prompts.prompt import PromptTemplate

def LLM(message):
    return Conversation.predict(input=message)

def Setup_TTS():
    global engine
    engine = pyttsx3.init()
    engine.setProperty('rate', 180)
    engine.setProperty('volume', 1)
    voice = engine.getProperty('voices')
    engine.setProperty('voice', voice[1].id)

def TTS(message):
    engine.say(message)
    engine.runAndWait()

def MicrophoneThread(audio_queue):
    r = speech_recognition.Recognizer()
    r.energy_threshold = 300
    r.pause_threshold = 0.8
    r.dynamic_energy_threshold = False

    with speech_recognition.Microphone(sample_rate=16000) as source:
        while True:
            audio = r.listen(source)

            torch_audio = torch.from_numpy(np.frombuffer(audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0)
            audio_data = torch_audio
            audio_queue.put_nowait(audio_data)

def LLM_Response(message):
    if(message.lower() == "And now exit.".lower()):
        os._exit(0)
        
    ai_response = LLM(message).lstrip()

    print(f"{AI_Name}: {ai_response}")
    TTS(ai_response)

def InputThread(text_queue):
    while True:
        text_input = input("").strip()
        text_queue.put_nowait(text_input)

def main():
    global LLM_Model
    global Conversation
    global AI_Name

    try:
        with open("config.json", "r") as json_file:
            data = json.load(json_file)
    except:
        print("Unable to open config")
        exit()
    

    Setup_TTS()

    AI_Name = data["AI_Name"]

    LLM_Model = OpenAI(
        temperature = 0.9,
        max_tokens = 256,
        top_p = 1,
        frequency_penalty = 1,
        presence_penalty = 1,
        openai_api_key = data["OAI_Key"],
        model_name = "text-davinci-003"
    )

    Conversation = ConversationChain(
        llm = LLM_Model,
        prompt=PromptTemplate(input_variables=["history", "input"], template= data["prompt"]),
        memory=ConversationSummaryMemory(llm = LLM_Model)
    )

    audio_model = whisper.load_model("base.en")
    audio_queue = queue.Queue()
    text_queue = queue.Queue()

    threading.Thread(target=MicrophoneThread, args=(audio_queue,)).start()
    threading.Thread(target=InputThread, args=(text_queue,)).start()

    while True:
        if not text_queue.empty():
            text_data = text_queue.get()
            LLM_Response(message=text_data)
        elif not audio_queue.empty():
            audio_data = audio_queue.get()
            result = audio_model.transcribe(audio_data,language="english")
            predicted_text = result["text"].lstrip()
            print(f"You: {predicted_text}")
            LLM_Response(message=predicted_text);
        else:
            pass

main()
"""def transcribe_audio(file_path: str) -> str:
    import whisper
    #try:
    model = whisper.load_model("base")
    res = model.transcribe(file_path)
    return res.get("text", "")
    #except Exception:
    return "error"

from faster_whisper import WhisperModel

def transcribe_audio(file_path):
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(file_path)
    return " ".join(segment.text for segment in segments)

print(transcribe_audio("uploads\b9fba493-351b-487e-8e3d-3180c9b52ef2_Sejal_2.m4a"))

from openai import OpenAI
client = OpenAI()

with open("uploads/b9fba493-351b-487e-8e3d-3180c9b52ef2_Sejal_2.m4a", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=f
    )
print(transcript.text)


trans = transcribe_audio('uploads\b9fba493-351b-487e-8e3d-3180c9b52ef2_Sejal_2.m4a')
print({trans})import speech_recognition as sr
"""
from pydub import AudioSegment
import speech_recognition as sr

def transcribe_audio_google(audio_file):
    sound = AudioSegment.from_file(audio_file)
    wav_path = "temp.wav"
    sound.export(wav_path, format="wav")

    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio_data = recognizer.record(source)
    return recognizer.recognize_google(audio_data)
    
"""from pydub import AudioSegment
import speech_recognition as sr

def transcribe_audio_google(audio_file):
    # Convert m4a to wav
    if audio_file.endswith(".m4a"):
        sound = AudioSegment.from_file(audio_file, format="m4a")
        wav_path = audio_file.replace(".m4a", ".wav")
        sound.export(wav_path, format="wav")
        audio_file = wav_path

    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
    return recognizer.recognize_google(audio_data)
"""
transcript_text = transcribe_audio_google(r"uploads\4dff9a69-5f74-480e-a418-28d1ebe561bc_Sejal_2_wav.wav")
#uploads\4dff9a69-5f74-480e-a418-28d1ebe561bc_Sejal_2_wav.wav
#C:\Users\lenovo\Documents\DebriefApp\uploads\4dff9a69-5f74-480e-a418-28d1ebe561bc_Sejal_2_wav.wav
print(transcript_text)
"""# Example
transcript_text = transcribe_audio_google(r"uploads\b9fba493-351b-487e-8e3d-3180c9b52ef2_Sejal_2.m4a")
print(transcript_text)

import speech_recognition as sr
def transcribe_audio_google(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Sorry, could not understand the audio."
    except sr.RequestError as e:
        return f"Google Speech Recognition request failed: {e}"
transcript_text = transcribe_audio_google('uploads\b9fba493-351b-487e-8e3d-3180c9b52ef2_Sejal_2.m4a')
"""


from gtts import gTTS
from playsound import playsound

#this is a fast hack
#TODO make it a module to be called whenever a new pose/person/whatever is added
tts = gTTS(text="Cayden Pierce", lang='en')
#tts.write_to_fp(audioAnswer)
tts.save('./CaydennPierce.mp3')

tts = gTTS(text="Abdallah Shami", lang='en')
#tts.write_to_fp(audioAnswer)
tts.save('./AbdalahShami.mp3')
#fill in text vars with the text you want to generate TTS

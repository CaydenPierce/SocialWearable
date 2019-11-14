from gtts import gTTS
from playsound import playsound

tts = gTTS(text="neck touch", lang='en')
#tts.write_to_fp(audioAnswer)
tts.save('./necktouch.mp3')

tts = gTTS(text="head touch", lang='en')
#tts.write_to_fp(audioAnswer)
tts.save('./headtouch.mp3')
#fill in text vars with the text you want to generate TTS

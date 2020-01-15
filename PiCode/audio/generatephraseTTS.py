from gtts import gTTS

def TTS(phrase):
	tts = gTTS(text=phrase, lang='en')
	tts.save('./{}.mp3'.format(phrase))

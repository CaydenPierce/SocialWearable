import os
from time import sleep
dir_name = os.path.dirname(os.path.realpath(__file__))

while True:
	os.system("mpg123 {}/necktouch.mp3".format(dir_name))
	sleep(5)

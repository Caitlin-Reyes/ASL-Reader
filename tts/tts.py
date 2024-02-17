import pyttsx3

engine = pyttsx3.init() # object creation
voices = engine.getProperty('voices')
engine.setProperty('voice',voices[1].id)
file = open("test.txt", "r")

while True:
    content = file.readline()
    if not content:
        break
    engine.setProperty('rate', 160)
    engine.say(content)
    engine.runAndWait()
file.close()

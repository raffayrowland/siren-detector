import random
import soundata
import shutil
import os

SOURCE = "audio"
DESTINATION = "data"

# Create directories
os.makedirs("models", exist_ok=True)
os.makedirs("weights", exist_ok=True)

for soundClass in ["positive", "negative"]:
    os.makedirs(f"{DESTINATION}/{soundClass}", exist_ok=True)

dataset = soundata.initialize('urbansound8k', data_home="urbansound8k")
# dataset.download()
# dataset.validate()
dataDict = dataset.load_clips()
shuffled = list(dataDict.items())
random.shuffle(shuffled)

# copy file into it's correct directory
negativeCounter = 0
for key, clip in shuffled:
    soundClass = "positive" if clip.class_id == 8 else "negative"
    if soundClass == "negative" and negativeCounter < 929:
        shutil.copy(clip.audio_path, f"{DESTINATION}/{soundClass}/{key}.wav")
        print(f"Copied file {key}.wav to {DESTINATION}/{soundClass}/{key}.wav")
        negativeCounter += 1

    elif soundClass == "positive":
        shutil.copy(clip.audio_path, f"{DESTINATION}/{soundClass}/{key}.wav")
        print(f"Copied file {key}.wav to {DESTINATION}/{soundClass}/{key}.wav")

print("Generated and populated class directories")
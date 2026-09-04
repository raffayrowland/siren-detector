import urllib.request
import soundata
import soundfile
import shutil
import os

MIN_DURATION_SECONDS = 2.0

# Create directories
os.makedirs("models", exist_ok=True)

for fold in range(1, 11):
    for soundClass in ["positive", "negative"]:
        os.makedirs(f"data/fold_{fold}/{soundClass}", exist_ok=True)

dataset = soundata.initialize('urbansound8k', data_home="urbansound8k")
dataset.download()
dataset.validate()
dataDict = dataset.load_clips()
items = list(dataDict.items())

# copy file into it's correct directory
for key, clip in items:
    destination = f"data/fold_{str(clip.fold)}/{"positive" if clip.class_id == 8 else "negative"}/{key}.wav"

    if soundfile.info(clip.audio_path).duration >= MIN_DURATION_SECONDS:
        shutil.copy(clip.audio_path, destination)
        print(f"Copied file {key}.wav to {destination}")

print("Generated and populated class directories")

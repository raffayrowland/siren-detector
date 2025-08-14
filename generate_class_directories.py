import soundata
import shutil
import os

SOURCE = "audio"
DESTINATION = "data"

# Create directories

for soundClass in ["positive", "negative"]:
    os.makedirs(f"{DESTINATION}/{soundClass}", exist_ok=True)

dataset = soundata.initialize('urbansound8k', data_home="C:\\Users\\raffa\\PycharmProjects\\siren-detector")
dataset.download()
dataset.verify()
dataDict = dataset.load_clips()

# copy file into it's correct directory
for key, clip in dataDict.items():
    soundClass = "positive" if clip.class_id == 8 else "negative"
    shutil.copy(clip.audio_path, f"{DESTINATION}/{soundClass}/{key}.wav")
    print(f"Copied file {key}.wav to {DESTINATION}/{soundClass}/{key}.wav")

print("Generated and populated class directories")
import soundata
import shutil
import os

SOURCE = "audio"
DESTINATION = "data"

# Create directories
for subset in ["train", "val", "test"]:
    for soundClass in ["positive", "negative"]:
        os.makedirs(f"{DESTINATION}/{subset}/{soundClass}", exist_ok=True)

dataset = soundata.initialize('urbansound8k', data_home="C:\\Users\\raffa\\PycharmProjects\\siren-detector")
dataDict = dataset.load_clips()

# Map folds 1-8 to train, 9 to val, and 10 to test
fold2subset = {**{f: "train" for f in range(1, 9)}, 9: "val", 10: "test"}

# copy file into it's correct directory
for key, clip in dataDict.items():
    subset = fold2subset[clip.fold]
    soundClass = "positive" if clip.class_id == 8 else "negative"
    shutil.copy(clip.audio_path, f"{DESTINATION}/{subset}/{soundClass}/{key}.wav")
    print(f"Copied file {key}.wav to {DESTINATION}/{subset}/{soundClass}/{key}.wav")

print("Generated and populated class directories")
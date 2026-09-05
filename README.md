# Siren Detector

Detects emergency vehicle sirens in real time using machine learning to assist people who are hard of hearing while driving or walking near roads. It could also be implemented in any car as an extra safety layer for all drivers

## Demonstrations

None of the audio used in the demonstration is in the training dataset

Demo with siren present:

[![Siren present demo](https://img.youtube.com/vi/eS7ugxmtGb0/0.jpg)](https://www.youtube.com/watch?v=eS7ugxmtGb0)

Demo with no siren present:

[![No siren demo](https://img.youtube.com/vi/wHFWJi1kZ6E/0.jpg)](https://www.youtube.com/watch?v=wHFWJi1kZ6E)

## Approach

The previous 2 seconds of audio is encoded using yamnet which produces a sequence of embeddings, and their mean is 
concatenated into a 2048-dimensional embedding. This embedding is provided to the trained head, which outputs a number 
from 0 to 1. If that number is over the threshold, the sample is deemed positive. For a "siren present" output to be 
shown to a user, the previous 10 predictions need to be positive. This is to reduce the effect of false positives.

## Installation

Clone the repository and set up the environment:

```
git clone https://github.com/raffayrowland/siren-detector.git
cd siren-detector
pip install -r requirements.txt
python build_requirements.py
python train.py
python predict_live.py
````

## Usage

* Detects sirens in real time from microphone input
* Can be integrated into cars to display visual warnings
* Useful for drivers with hearing impairments and for added safety in general

## Tech stack

* Python 3.12
* Tensorflow
* yamnet encoder
* numpy

## Dataset

This project uses UrbanSound8K, using clips tagged 'siren' as positive samples and everything else as negative. 
clips shorter than 2 seconds are not used
* [UrbanSound8k Dataset](https://urbansounddataset.weebly.com/urbansound8k.html)


## Evaluation

* Precision: 0.98 on test set
* Recall: 0.93 on test set

## License

**Code**: MIT License – freely use, modify, distribute, including commercially.

**Dataset (UrbanSound8K)**: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0). Use only for non-commercial purposes. You must attribute the dataset creators:

> Salamon, J., Jacoby, C., & Bello, J. P. (2014). *A Dataset and Taxonomy for Urban Sound Research*. 22nd ACM Int. Conf. on Multimedia.

For dataset terms, see: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).


# Siren Detector

Detects emergency vehicle sirens in real time using machine learning to assist people who are hard of hearing while driving or walking near roads. It could also be implemented in any car as an extra safety layer for all drivers

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

## Requirements

* Python 3.x
* 4GB RAM for training
* 1GB RAM for inference
* 550mb disk space for model
* Packages in `requirements.txt`

## Tech stack

* Python 3.11
* Tensorflow
* CNN on log-mel spectrograms

## Dataset

* Used UrbanSound8k 
* Preprocessing: converted clips to log-mel spectrograms
* [UrbanSound8k Dataset](https://urbansounddataset.weebly.com/urbansound8k.html)


## Evaluation

* Accuracy: 96% on test set
* Precision: 0.94
* Recall: 0.92

## License

**Code**: MIT License – freely use, modify, distribute, including commercially.

**Dataset (UrbanSound8K)**: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0). Use only for non-commercial purposes. You must attribute the dataset creators:

> Salamon, J., Jacoby, C., & Bello, J. P. (2014). *A Dataset and Taxonomy for Urban Sound Research*. 22nd ACM Int. Conf. on Multimedia.

For dataset terms, see: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).


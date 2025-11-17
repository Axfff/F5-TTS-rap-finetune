# SpeechScore

## 👉🏻[HuggingFace Space Demo](https://huggingface.co/spaces/alibabasglab/SpeechScore)👈🏻

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Usage](#2-usage)
- [3. Acknowledgements](#3-acknowledgements)

## 1. Introduction

SpeechScore is a wrapper designed for assessing speech quality. It includes a collection of commonly used speech quality metrics, as listed below:
| Index | Metrics | Short Description | Externel Link |
|-------|---------|-------------|---------------|
|1.| BSSEval {ISR, SAR, SDR} | ISR (Source Image-to-Spatial distortion Ratio) measures preservation/distortion of target source. SDR (Source-to-Distortion Ratio) measures global quality. SAR (Source-to-Artefact Ratio) measures the presence of additional artificial noise|(See <a href="https://github.com/sigsep/sigsep-mus-eval">the official museval page</a>)|
|2.| {CBAK, COVL, CSIG} | CSIG predicts the signal distortion mean opinion score (MOS), CBAK measures background intrusiveness, and COVL measures speech quality. CSIG, CBAK, and COVL are ranged from 1 to 5| See paper: <a href="https://ecs.utdallas.edu/loizou/speech/obj_paper_jan08.pdf">Evaluation of Objective Quality Measures for Speech Enhancement</a>|
|3.| DNSMOS {BAK, OVRL, SIG, P808_MOS} |DNSMOS (Deep Noise Suppression Mean Opinion Score) measures the overall quality of the audio clip based on the ITU-T Rec. P.808 subjective evaluation. It outputs 4 scores: i) speech quality (SIG), ii) background noise quality (BAK), iii) the overall quality (OVRL), and iv) the P808_MOS of the audio.  DNSMOS does not require clean references. | See paper: <a href="https://arxiv.org/pdf/2010.15258.pdf">Dnsmos: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors</a> and <a href="https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS">github page</a>|
|4.| FWSEGSNR | FWSEGSNR (Frequency-Weighted SEGmental SNR) is commonly used for evaluating dereverberation performance |See paper: <a href="https://ecs.utdallas.edu/loizou/speech/obj_paper_jan08.pdf">Evaluation of Objective Quality Measures for Speech Enhancement</a> | 
|5.| LLR |LLR (Log Likelihood Ratio) measures how well an estimated speech signal matches the target (clean) signal in terms of their short-term spectral characteristics. |See paper: <a href="https://ecs.utdallas.edu/loizou/speech/obj_paper_jan08.pdf">Evaluation of Objective Quality Measures for Speech Enhancement</a> |
|6.| LSD | LSD (Log-Spectral Distance) measures the spectral differences between a clean reference signal and a processed speech signal.| See <a href="https://github.com/haoheliu/ssr_eval"> github page </a>|
|7.| MCD | MCD (Mel-Cepstral Distortion) measures the difference between the mel-cepstral coefficients (MCCs) of an estimated speech signal and the target (clean) speech signal. |See <a href="https://github.com/chenqi008/pymcd"> github page </a> |
|8.| NB_PESQ |NB-PESQ (NarrowBand Perceptual Evaluation of Speech Quality) meaures speech quality that reflects human auditory perception. It is defined in the ITU-T Recommendation P.862 and is developed for assessing narrowband speech codecs and enhancement algorithms. | See <a href="https://github.com/ludlows/PESQ"> github page </a> |
|9.| PESQ | PESQ (Perceptual Evaluation of Speech Quality) assesses the quality of speech signals to mimic human perception. It is standardized by the International Telecommunication Union (ITU-T P.862) and is widely used in evaluating telecommunication systems and speech enhancement algorithms. |See <a href="https://github.com/ludlows/PESQ"> github page </a> |
|10.| SISDR |SI-SDR (Scale-Invariant Signal-to-Distortion Ratio) quantifies the ratio between the power of the target signal component and the residual distortion. It measures how well an estimated speech signal matches the target (clean) speech signal, while being invariant to differences in scale. |See paper: <a href="https://arxiv.org/abs/1811.02508">SDR - half-baked or well done?<a/> |
|11.| SNR | SNR (Signal-to-Noise Ratio) is a fundamental metric used in speech quality measurement to evaluate the relative level of the desired speech signal compared to unwanted noise. It quantifies the clarity and intelligibility of speech in decibels (dB).| See paper: <a href="https://www.isca-archive.org/icslp_1998/hansen98_icslp.pdf">An effective quality evaluation protocol for speech enhancement algorithms<a/>|
|12.| SRMR |SRMR (Speech-to-Reverberation Modulation Energy Ratio) evaluates the ratio of speech-dominant modulation energy to reverberation-dominant modulation energy. It quantifies the impact of reverberation on the quality and intelligibility of speech signals. SRMR does not require clean references. | See <a href="https://github.com/jfsantos/SRMRpy">SRMRpy<a/> and <a href="https://github.com/MuSAELab/SRMRToolbox">SRMR Toolbox<a/>|
|13.| SSNR |SSNR (Segmental Signal-to-Noise Ratio) is an extension of SNR (Signal-to-Noise Ratio) and for evaluating the quality of speech signals in shorter segments or frames. It is calculated by dividing the power of the clean speech signal by the power of the noise signal, computed over small segments of the speech signal. | See paper: <a href="https://www.isca-archive.org/icslp_1998/hansen98_icslp.pdf">An effective quality evaluation protocol for speech enhancement algorithms<a/>|
|14.| STOI|STOI (Short-Time Objective Intelligibility Index) measures speech quality and intelligibility by operateing on short-time segments of the speech signal and computes a score between 0 and 1. | See <a href="https://github.com/mpariente/pystoi">github page <a/> |

## 2. Usage

### Step-by-Step Guide

If you haven't created a Conda environment for ClearerVoice-Studio yet, follow steps 1 and 2. Otherwise, skip directly to step 3.

1. **Clone the Repository**

``` sh
git clone https://github.com/modelscope/ClearerVoice-Studio.git
```

2. **Create Conda Environment**

``` sh
cd ClearerVoice-Studio
conda create -n ClearerVoice-Studio python=3.8
conda activate ClearerVoice-Studio
pip install -r requirements.txt
```

3. Run demo script

``` sh
cd speechscore
python demo.py
```
or use the following script:
``` python
# Import pprint for pretty-printing the results in a more readable format
import pprint
# Import the SpeechScore class to evaluate speech quality metrics
from speechscore import SpeechScore 

# Main block to ensure the code runs only when executed directly
if __name__ == '__main__':
    # Initialize a SpeechScore object with a list of score metrics to be evaluated
    # Supports any subsets of the list
    mySpeechScore = SpeechScore([
        'SRMR', 'PESQ', 'NB_PESQ', 'STOI', 'SISDR', 
        'FWSEGSNR', 'LSD', 'BSSEval', 'DNSMOS', 
        'SNR', 'SSNR', 'LLR', 'CSIG', 'CBAK', 
        'COVL', 'MCD'
    ])

    # Call the SpeechScore object to evaluate the speech metrics between 'noisy' and 'clean' audio
    # Arguments:
    # - {test_path, reference_path} supports audio directories or audio paths (.wav or .flac)
    # - window (float): seconds, set None to specify no windowing (process the full audio)
    # - score_rate (int): specifies the sampling rate at which the metrics should be computed
    # - return_mean (bool): set True to specify that the mean score for each metric should be returned

    
    print('score for a signle wav file')
    scores = mySpeechScore(test_path='audios/noisy.wav', reference_path='audios/clean.wav', window=None, score_rate=16000, return_mean=False)
    # Pretty-print the resulting scores in a readable format
    pprint.pprint(scores)

    print('score for wav directories')
    scores = mySpeechScore(test_path='audios/noisy/', reference_path='audios/clean/', window=None, score_rate=16000, return_mean=True)

    # Pretty-print the resulting scores in a readable format
    pprint.pprint(scores)

    # Print only the resulting mean scores in a readable format
    #pprint.pprint(scores['Mean_Score'])
```
The results should be looking like below:

```sh
score for a signle wav file
{'BSSEval': {'ISR': 22.74466768594831,
             'SAR': -0.1921607960486258,
             'SDR': -0.23921670199308115},
 'CBAK': 1.5908301020179343,
 'COVL': 1.5702204013203889,
 'CSIG': 2.3259366746377066,
 'DNSMOS': {'BAK': 1.3532928733331306,
            'OVRL': 1.3714771994335782,
            'P808_MOS': 2.354834,
            'SIG': 1.8698058813241407},
 'FWSEGSNR': 6.414399025759913,
 'LLR': 0.85330075,
 'LSD': 2.136734818644327,
 'MCD': 11.013451521306235,
 'NB_PESQ': 1.2447538375854492,
 'PESQ': 1.0545592308044434,
 'SISDR': -0.23707451176264824,
 'SNR': -0.9504614142497447,
 'SRMR': 6.202590182397157,
 'SSNR': -0.6363067113236048,
 'STOI': 0.8003376411051097}
 
score for wav directories
{'Mean_Score': {'BSSEval': {'ISR': 23.728811184378372,
                            'SAR': 4.839625092004951,
                            'SDR': 4.9270216975279135},
                'CBAK': 1.9391528046230797,
                'COVL': 1.5400270840455588,
                'CSIG': 2.1286157747587344,
                'DNSMOS': {'BAK': 1.9004402577440938,
                           'OVRL': 1.860621534493506,
                           'P808_MOS': 2.5821499824523926,
                           'SIG': 2.679913397827385},
                'FWSEGSNR': 9.079539440199582,
                'LLR': 1.1992616951465607,
                'LSD': 2.0045290996104748,
                'MCD': 8.916492705343465,
                'NB_PESQ': 1.431145429611206,
                'PESQ': 1.141619324684143,
                'SISDR': 4.778657656271212,
                'SNR': 4.571920494312266,
                'SRMR': 9.221118316293268,
                'SSNR': 2.9965604574762796,
                'STOI': 0.8585249663711918},
 'audio_1.wav': {'BSSEval': {'ISR': 22.74466768594831,
                             'SAR': -0.1921607960486258,
                             'SDR': -0.23921670199308115},
                 'CBAK': 1.5908301020179343,
                 'COVL': 1.5702204013203889,
                 'CSIG': 2.3259366746377066,
                 'DNSMOS': {'BAK': 1.3532928733331306,
                            'OVRL': 1.3714771994335782,
                            'P808_MOS': 2.354834,
                            'SIG': 1.8698058813241407},
                 'FWSEGSNR': 6.414399025759913,
                 'LLR': 0.85330075,
                 'LSD': 2.136734818644327,
                 'MCD': 11.013451521306235,
                 'NB_PESQ': 1.2447538375854492,
                 'PESQ': 1.0545592308044434,
                 'SISDR': -0.23707451176264824,
                 'SNR': -0.9504614142497447,
                 'SRMR': 6.202590182397157,
                 'SSNR': -0.6363067113236048,
                 'STOI': 0.8003376411051097},
 'audio_2.wav': {'BSSEval': {'ISR': 24.712954682808437,
                             'SAR': 9.871410980058528,
                             'SDR': 10.093260097048908},
                 'CBAK': 2.287475507228225,
                 'COVL': 1.509833766770729,
                 'CSIG': 1.9312948748797627,
                 'DNSMOS': {'BAK': 2.4475876421550566,
                            'OVRL': 2.349765869553434,
                            'P808_MOS': 2.809466,
                            'SIG': 3.490020914330629},
                 'FWSEGSNR': 11.744679854639253,
                 'LLR': 1.5452226,
                 'LSD': 1.8723233805766222,
                 'MCD': 6.819533889380694,
                 'NB_PESQ': 1.617537021636963,
                 'PESQ': 1.2286794185638428,
                 'SISDR': 9.794389824305073,
                 'SNR': 10.094302402874277,
                 'SRMR': 12.23964645018938,
                 'SSNR': 6.629427626276164,
                 'STOI': 0.9167122916372739}}
```
Any subset of the full score list is supported, specify your score list using the following objective:

```
mySpeechScore = SpeechScore(['.'])
```

## 3. Acknowledgements
We referred to <a href="https://github.com/aliutkus/speechmetrics">speechmetrics<a/>, <a href="https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS">DNSMOS <a/>, <a href="https://github.com/sigsep/bsseval/tree/master">BSSEval<a/>, <a href="https://github.com/chenqi008/pymcd/blob/main/pymcd/mcd.py">pymcd<a/>, <a href="https://github.com/mpariente/pystoi">pystoi<a/>, <a href="https://github.com/ludlows/PESQ">PESQ<a/>, and <a href="https://github.com/santi-pdp/segan_pytorch/tree/master">segan_pytorch<a/> for implementing this repository.

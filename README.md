# Edge-tts with DDSP-SVC



## 简介 / Introduction

本仓库基于edge-tts与DDSP-SVC，简单实现了文本控制语音生成与音色转换技术的结合。

This repo is based on edge-tts and DDSP-SVC, and achieves a simple combination of text to speech and singing voice conversion.



## 安装依赖 / Installing the dependencies

在DDSP-SVC环境的基础上，只需安装edge-tts。

On the basis of DDSP-SVC environment, only edge-tts needs to be installed.

```
$ pip install edge-tts==6.1.3
```



## 使用方法 / Usage

### step 1

将文件移动至DDSP-SVC仓库主目录下。

Move the files to the main directory of the DDSP-SVC repository.

### step 2

在start.py文件中修改下列参数：

- **TEXT_INPUT_PATH**，存放待转换文本的txt文件路径；

- **VOICE**，edge-tts说话人；

- **OUTPUT_NAME**，输出音频文件的名称；

- **MODEL_PATH**，DDSP-SVC模型路径，确保**config.yaml**在同一文件夹内；

- **KEY_CHANGE**，升降调（半音）；

- **PITCH_EXTRACTOR**，PITCH_EXTRACTOR的选择。

  

Modify the following parameters in main.py: 

- **TEXT_INPUT_PATH**, the path to the txt file that stores the text to be converted; 
- **VOICE**, the edge-tts speaker; 
- **OUTPUT_NAME**, the file name of the output audio;
- **MODEL_PATH**, the path of the DDSP-SVC model, make sure that **config.yaml** is in the same folder;
- **KEY_CHANGE**, semitones change;
- **PITCH_EXTRACTOR**, pitch extrator type: parselmouth, dio, harvest, crepe.

```python
# Modifiable
TEXT_INPUT_PATH = "./text.txt"  # Path of text to be converted
VOICE = "zh-CN-XiaoyiNeural"  # You can change the available voice here
OUTPUT_NAME = "output"  # Name of the output file
MODEL_PATH = "./model_best.pt"  # Model path (make sure that config.yaml is in the same folder)
KEY_CHANGE = 0  # Key change (number of semitones)
PITCH_EXTRACTOR = 'crepe'  # Pitch extrator type: parselmouth, dio, harvest, crepe
```

### step 3

在**TEXT_INPUT_PATH**指向的txt文件中填写待转换文本。换行（回车键）代表换段，每段单独生成一个音频文件。

Place the text to be converted in the txt file of **TEXT_INPUT_PATH**. A newline (Enter key) means a new segment, and each segment can generate an independent audio.

### step 4

运行start.py文件。

Run start.py.

```
$ python start.py
```



## 其它 / Others

如果音色转换的效果不理想，可以在start.py中修改f0和threhold的参数。具体修改策略请参考DDSP-SVC仓库的教程文档。

If the effect of voice conversion does not meet your expectations, you can modify the parameters of f0 and threhold in the start.py. For modification strategies, please refer to the tutorial documentation of DDSP-SVC repository.



## 参考 / Reference

[yxlllc/DDSP-SVC: Real-time end-to-end singing voice conversion system based on DDSP (Differentiable Digital Signal Processing) (github.com)](https://github.com/yxlllc/DDSP-SVC)

[rany2/edge-tts: Use Microsoft Edge's online text-to-speech service from Python (without needing Microsoft Edge/Windows or an API key) (github.com)](https://github.com/rany2/edge-tts)

import os
from functions import generate_wav, split, cross_fade
from ddsp.vocoder import load_model, F0_Extractor, Volume_Extractor, Units_Encoder
from ddsp.core import upsample
import logging
import torch
import librosa
import argparse
import numpy as np
import soundfile as sf
import pyworld as pw
import parselmouth
from ast import literal_eval
from slicer import Slicer
from enhancer import Enhancer
from tqdm import tqdm

# Modifiable
TEXT_INPUT_PATH = "./text.txt"  # Path of text to be converted
VOICE = "zh-CN-XiaoyiNeural"  # You can change the available voice here
OUTPUT_NAME = "output"  # Name of the output file
MODEL_PATH = "./model_best.pt"  # Model path (make sure that config.yaml is in the same folder)
KEY_CHANGE = 0  # Key change (number of semitones)
PITCH_EXTRACTOR = 'crepe'  # Pitch extrator type: parselmouth, dio, harvest, crepe


# Unmodifiable
OUTPUT_PATH = "./output_tts/"
F0_MIN = 50  # min f0 (Hz)
F0_MAX = 1100  # max f0 (Hz)
THREHOLD = -60  # response threhold (dB)


if __name__ == "__main__":
    logging.getLogger('numba').setLevel(logging.WARNING)

    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    text = open(TEXT_INPUT_PATH, encoding="utf-8")
    lines = text.readlines()  # Line feed means segmentation, each segment generates an audio file
    for i, line in enumerate(lines):
        output_file = OUTPUT_PATH + OUTPUT_NAME + "_ori_" + str(i) + ".wav"
        generate_wav(line, VOICE, output_file)
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, args = load_model(MODEL_PATH, device=device)
    
    units_encoder = Units_Encoder(
                        args.data.encoder, 
                        args.data.encoder_ckpt, 
                        args.data.encoder_sample_rate, 
                        args.data.encoder_hop_size, 
                        device = device)
    
    for i in range(len(lines)):
        input_wav_file = OUTPUT_PATH + OUTPUT_NAME + "_ori_" + str(i) + ".wav"
        while not os.path.exists(input_wav_file):
            print("Waiting for file generation")
            
        audio, sample_rate = librosa.load(input_wav_file, sr=None) 
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        hop_size = args.data.block_size * sample_rate / args.data.sampling_rate
        
        print('Pitch extractor type: ' + PITCH_EXTRACTOR)
        pitch_extractor = F0_Extractor(
                            PITCH_EXTRACTOR, 
                            sample_rate, 
                            hop_size, 
                            float(F0_MIN), 
                            float(F0_MAX))
        print('Extracting the pitch curve of the input audio...')
        f0 = pitch_extractor.extract(audio, uv_interp = True, device = device)
        f0 = torch.from_numpy(f0).float().to(device).unsqueeze(-1).unsqueeze(0)
        
        f0 = f0 * 2 ** (float(KEY_CHANGE) / 12)
        
        print('Extracting the volume envelope of the input audio...')
        volume_extractor = Volume_Extractor(hop_size)
        volume = volume_extractor.extract(audio)
        mask = (volume > 10 ** (float(THREHOLD) / 20)).astype('float')
        mask = np.pad(mask, (4, 4), constant_values=(mask[0], mask[-1]))
        mask = np.array([np.max(mask[n : n + 9]) for n in range(len(mask) - 8)])
        mask = torch.from_numpy(mask).float().to(device).unsqueeze(-1).unsqueeze(0)
        mask = upsample(mask, args.data.block_size).squeeze(-1)
        volume = torch.from_numpy(volume).float().to(device).unsqueeze(-1).unsqueeze(0)
        
        spk_mix_dict = literal_eval("None")
        spk_id = torch.LongTensor(np.array([[1]])).to(device)
        
        result = np.zeros(0)
        current_length = 0
        segments = split(audio, sample_rate, hop_size)
        print('Cut the input audio into ' + str(len(segments)) + ' slices')
        with torch.no_grad():
            for segment in tqdm(segments):
                start_frame = segment[0]
                seg_input = torch.from_numpy(segment[1]).float().unsqueeze(0).to(device)
                seg_units = units_encoder.encode(seg_input, sample_rate, hop_size)

                seg_f0 = f0[:, start_frame : start_frame + seg_units.size(1), :]
                seg_volume = volume[:, start_frame : start_frame + seg_units.size(1), :]

                seg_output, _, (s_h, s_n) = model(seg_units, seg_f0, seg_volume, spk_id = spk_id, spk_mix_dict = spk_mix_dict)
                seg_output *= mask[:, start_frame * args.data.block_size : (start_frame + seg_units.size(1)) * args.data.block_size]

                output_sample_rate = args.data.sampling_rate

                seg_output = seg_output.squeeze().cpu().numpy()

                silent_length = round(start_frame * args.data.block_size * output_sample_rate / args.data.sampling_rate) - current_length
                if silent_length >= 0:
                    result = np.append(result, np.zeros(silent_length))
                    result = np.append(result, seg_output)
                else:
                    result = cross_fade(result, seg_output, current_length + silent_length)
                current_length = current_length + silent_length + len(seg_output)
                
            wav_gen = OUTPUT_PATH + OUTPUT_NAME + "_target_" + str(i) + ".wav"
            sf.write(wav_gen, result, output_sample_rate)

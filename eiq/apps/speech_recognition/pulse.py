# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import wave as _wave

import numpy as _np
import pygal

def _convert_wav_file_to_array(nchannels, sampwidth, data: str = None):
    num_samples, remainder = divmod(len(data), sampwidth * nchannels)
    if (remainder > 0):
        raise ValueError('The length of data is not a multiple of '
                         'sampwidth * num_channels.')
    if (sampwidth > 4):
        raise ValueError("The sample width must not be greater than 4.")

    if (sampwidth == 3):
        a = _np.empty((num_samples, nchannels, 4), dtype=_np.uint8)
        raw_bytes = _np.fromstring(data, dtype=_np.uint8)
        a[:, :, :sampwidth] = raw_bytes.reshape(-1, nchannels, sampwidth)
        a[:, :, sampwidth:] = (a[:, :, sampwidth - 1:sampwidth] >> 7) * 255
        result = a.view('<i4').reshape(a.shape[:-1])
    else:
        # 8 bit samples are stored as unsigned ints; others as signed ints.
        dt_char = 'u' if sampwidth == 1 else 'i'
        a = _np.fromstring(data, dtype='<%s%d' % (dt_char, sampwidth))
        result = a.reshape(-1, nchannels)
    return result

def read_wav_file(file):
    # This function does not read compressed WAV files.
    try:
        wav_file = _wave.open(file)
    except:
        raise ValueError("Could not open wav file properly\n")

    # sampling frequency
    wav_file_rate = wav_file.getframerate()
    print("Sampling Frequency: " + str(wav_file_rate))
    
    # 1 for mono, 2 for stereo
    wav_file_nchannels = wav_file.getnchannels()
    channel = "mono" if wav_file_nchannels == 1 else "stereo"
    print("Number of Channels: " + str(wav_file_nchannels) + " " + channel)
    
    # (float) Sample width in bytes. E.g. for a 24 bit WAV file, sampwidth is 3.
    wav_file_sample_width = wav_file.getsampwidth()
    print("Sample Width in Bytes: " + str(wav_file_sample_width))
    
    # number of audio frames
    wav_file_nframes = wav_file.getnframes()
    print("Number of Audio Frames: " + str(wav_file_nframes))
    
    wav_file_data = wav_file.readframes(wav_file_nframes)
    wav_file.close()
        
    wav_file_data_array = _convert_wav_file_to_array(wav_file_nchannels,
                                                     wav_file_sample_width,
                                                     wav_file_data)
 
    duration = len(wav_file_data_array)/wav_file_rate
    wav_file_time = _np.arange(0, duration, 1/wav_file_rate) 
    
    return wav_file_rate, wav_file_sample_width, wav_file_data_array, wav_file_time

def generate_pulse_plot(wav_file_path):
    wav_rate, wav_width, wav_data, wav_time = read_wav_file(wav_file_path)

    # Needs modification to work faster.
    date_chart = pygal.Line()    
    an_array = _np.array(wav_data, dtype=_np.int16)
    wav_data = an_array.astype(_np.float)    
    
    date_chart.add(wav_file_path, wav_data)
    date_chart.render_to_png("pulse.png")

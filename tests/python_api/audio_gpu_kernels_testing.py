
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import numpy as np
# from amd.rocal.plugin.pytorch import ROCALClassificationIterator
from amd.rocal.plugin.pytorch import ROCALAudioIterator

import torch
# torch.set_printoptions(threshold=10_000)
np.set_printoptions(threshold=1000, edgeitems=10000)
from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types
import math
# import rocal_pybind.tensor
import sys
import cv2
import matplotlib.pyplot as plt
import os
def draw_patches(img, idx, device):
    #image is expected as a tensor, bboxes as numpy
    import cv2
    image = img.cpu().detach().numpy()
    audio_data = image.flatten()
    label = idx.cpu().detach().numpy()
    print("label: ", label)
    # Saving the array in a text file
    file = open("results/rocal_data_new"+str(label)+".txt", "w+")
    content = str(audio_data)
    file.write(content)
    file.close()
    plt.plot(audio_data)
    plt.savefig("results/rocal_data_new"+str(label)+".png")
    plt.close()
def main():
    if  len(sys.argv) < 3:
        print ('Please pass audio_folder file_list cpu/gpu batch_size')
        exit(0)
    try:
        path= "OUTPUT_IMAGES_PYTHON/NEW_API/FILE_READER/" + "audio"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    except OSError as error:
        print(error)
    data_path = sys.argv[1]
    file_list = sys.argv[2]
    if(sys.argv[3] == "cpu"):
        _rali_cpu = True
    else:
        _rali_cpu = False
    batch_size = int(sys.argv[4])
    num_threads = 1
    device_id = 0
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)
    crop=300
    local_rank = 0
    world_size = 1
    print("*********************************************************************")
    audio_pipeline = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=random_seed, rocal_cpu=_rali_cpu)
    with audio_pipeline:
        audio, label = fn.readers.file(
            # **files_arg,
            file_root=data_path,
            file_list=file_list,
            )
        sample_rate = 16000
        nfft=512
        window_size=0.02
        window_stride=0.01
        nfilter=80 #nfeatures
        resample = 16000.00
        # dither = 0.001
        # audio_decode = fn.decoders.audio(audio, file_root=data_path, downmix=True, shard_id=0, num_shards=2,random_shuffle=True)
        decoded_audio = fn.decoders.audio(
            audio, 
            file_root=data_path, 
            file_list_path=file_list, 
            downmix=True, 
            shard_id=0, 
            num_shards=2, 
            stick_to_shard=True,
            last_batch_policy=types.LAST_BATCH_FILL, pad_last_batch_repeated=False)
        # audio, labels = fn.readers.file(file_root=path, file_list=file_list)
        # decoded_audio = fn.decoders.audio(
        #     audio,
        #     file_root=path,
        #     file_list_path=file_list,
        #     downmix=True,
        #     shard_id=0,
        #     num_shards=2,
        #     stick_to_shard=True,
        #     last_batch_policy=types.LAST_BATCH_FILL, pad_last_batch_repeated=False)
        # uniform_distribution_resample = fn.random.uniform(decoded_audio, range=[0.8555555, 0.8555555])
        # resampled_rate = uniform_distribution_resample * resample
        # # # # # resample_output = fn.resample(audio_decode, resample_rate = resampled_rate, resample_hint=250000, )
        # resample_output = fn.resample(decoded_audio, resample_rate = resampled_rate, resample_hint=0.85555 * 258160, )
        begin, length = fn.nonsilent_region(decoded_audio, cutoff_db=-60)
        # trim_silence = fn.slice(
        #     decoded_audio,
        #     anchor=[begin],
        #     shape=[length],
        # )
        # normal_distribution = fn.random.normal(audio_decode, mean=0.0, stddev=0.0000001)
        # newAudio = normal_distribution * 0.00001
        # dist_audio = trim_silence + newAudio
        # premph_audio = fn.preemphasis_filter(decoded_audio)
        # spectrogram_audio = fn.spectrogram(
        #     premph_audio,
        #     nfft=nfft,
        #     window_length=320, # Change to 320
        #     window_step= 160, # Change to 160
        #     # rocal_tensor_output_type=types.FLOAT,
        # )
        # mel_filter_bank_audio = fn.mel_filter_bank(
        #     spectrogram_audio,
        #     sample_rate=sample_rate,
        #     nfilter=nfilter,
        # )
        # to_decibels_audio = fn.to_decibels(
        #     mel_filter_bank_audio,
        #     multiplier=math.log(10),
        #     reference=1.0,
        #     cutoff_db=math.log(1e-20),
        #     # rocal_tensor_output_type=types.FLOAT,
        # )
        # normalize_audio = fn.normalize(to_decibels_audio, axes=[1])
        # pad_audio = fn.pad(normalize_audio, fill_value=0)

        audio_pipeline.set_outputs(begin)
    audio_pipeline.build()
    audioIteratorPipeline = ROCALAudioIterator(audio_pipeline, auto_reset=True,device='cpu')
    cnt = 0
    for e in range(1):
        print("Epoch :: ", e)
        torch.set_printoptions(threshold=5000, profile="full", edgeitems=100)
        for i , it in enumerate(audioIteratorPipeline):
            print("************************************** i *************************************",i)
            print(it[2])
            for img, label, roi in zip(it[0],it[1],it[2]):
                # print("label", label)
                print("roi", roi)
                # print("img",img)
                # print(type(img))
                print(img.device)
                # exit(0)
                draw_patches(img, label, "cpu")
        print("EPOCH DONE", e)
if __name__ == '__main__':
    main()

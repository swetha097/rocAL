# rocAL Audio Unit Tests
This application can be used to verify the functionality of the Audio APIs offered by rocAL.

## Build Instructions

### Pre-requisites
* Ubuntu Linux, [version `20.04` or later](https://www.microsoft.com/software-download/windows10)
* rocAL library (Part of the MIVisionX toolkit)
* Radeon Performance Primitives (RPP)
* MIVisionX
* Sndfile

### Build
  ````
  mkdir build
  cd build
  cmake ../
  make
  ````
### Running the application
  ````
./rocal_audio_unittests <audio-dataset-folder>

Usage: ./rocal_audio_unittests <audio-dataset-folder> <test_case> <downmix> <device-gpu=1/cpu=0>
  ````

### Output verification 

The python script `rocal_audio_unittest.py` can be used to run all test cases for audio functionality in rocAL and verify the correctness of the generated outputs with the golden outputs.

Input data is available in the following link : [MIVisionX-data](https://github.com/ROCm/MIVisionX-data/rocal_data)

`export ROCAL_DATA_PATH=<absolute_path_to_rocal_data>`

```
python3 rocal_audio_unittest.py <device_type 0/1> <downmix 0/1>
```

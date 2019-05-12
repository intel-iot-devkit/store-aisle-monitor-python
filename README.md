

# Store Aisle Monitor


| Details               |                  |
|-----------------------|------------------|
| Target OS             |  Ubuntu\* 16.04 LTS     |
| Programming Language  |  Python* 3.5 |
| Time to complete      |  30 min      |

This reference implementation is also [available in C++](https://github.com/intel-iot-devkit/reference-implementation-private/blob/master/store-aisle-monitor/README.md)

## Introduction

This reference implementation counts the number of people present in an image and generates a motion heatmap. It takes the input from the camera, or a video file for processing. Snapshots of the output are taken at regular intervals and are uploaded to the cloud. It also stores the snapshots of the output locally.

## Requirements

### Hardware
*  6th to 8th generation Intel® Core™ processor with Intel® Iris® Pro graphics or Intel® HD Graphics

### Software

* [Ubuntu* 16.04](http://releases.ubuntu.com/16.04/)
* OpenCL™ Runtime Package<br>
  **Note**: We recommend using a 4.14+ kernel to use this software. Run the following command to determine your kernel version:
  ```  
  uname -a
  ```  
* Intel® Distribution of OpenVINO™ toolkit 2019 R1 Release
* Microsoft Azure* Python SDK

## Setup

### Install Intel® Distribution of OpenVINO™ toolkit

Refer to [https://software.intel.com/en-us/articles/OpenVINO-Install-Linux](https://software.intel.com/en-us/articles/OpenVINO-Install-Linux) for more information on how to install and setup the Intel® Distribution of OpenVINO™ toolkit.

The OpenCL™ Runtime package is required to run the inference on a GPU. It is not mandatory for CPU inference.

### Install Python* dependencies
```
sudo apt-get install python3-pip
pip3 install azure numpy
```

## How it Works
- The application uses a video source, such as a camera or a video file, to grab the frames. The [OpenCV functions](https://docs.opencv.org/3.4/dd/d43/tutorial_py_video_display.html) are used to calculate frame width, frame height and frames per second (fps) of the video source. The application counts the number of people and generates motion heatmap.
![Architectural diagram](./images/figure1.png)

-  People counter: A trained neural network model detects the people in the frame and bounding boxes are drawn on the people detected. This reference implementation uses a pre-trained model **person-detection-retail-0013** that can be downloaded using the **model downloader**, provided by the Intel® Distribution of OpenVINO™ toolkit.  

- Motion Heatmap generation: An accumulated frame is used, on which every frame is added after preprocessing. This accumulated frame is used to generate the motion heatmap using [applyColorMap](https://docs.opencv.org/3.4/d3/d50/group__imgproc__colormap.html#gadf478a5e5ff49d8aa24e726ea6f65d15). The original frame and heatmap frame are merged using [addWeighted](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_core/py_image_arithmetics/py_image_arithmetics.html), to visualize the movement patterns over time.

-  The heatmap frame and people counter frame are merged using [addWeighted](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_core/py_image_arithmetics/py_image_arithmetics.html) and this merged frame is saved locally at regular intervals. The output is present in the *output_snapshots* directory of the project directory.

    ![Output snapshot](./images/figure2.png)

-  The application also uploads the results to the Microsoft Azure cloud at regular intervals, if a Microsoft Azure storage name and key are provided. 
    ![Uploading snapshots to cloud](./images/figure3.png)

## Download the model

This application uses the **person-detection-retail-0013** Intel® pre-trained model, that can be accessed using the **model downloader**. The **model downloader** downloads the __.xml__ and __.bin__ files that will be used by the application.

Steps to download .xml and .bin files:<br>

* Go to the **model downloader** directory using following command:

      cd /opt/intel/openvino/deployment_tools/tools/model_downloader

* Specify which model to download with `--name`.<br>
  To download the **person-detection-retail-0013** model, run the following command:

      sudo ./downloader.py --name person-detection-retail-0013

* To download the **person-detection-retail-0013** model for **FP16**, run the following command:

      sudo ./downloader.py --name person-detection-retail-0013-fp16
 
The model will be downloaded inside the following directory:
 
    /opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_detection/pedestrian/rmnet_ssd/0013/dldt/

## Setup the environment

You must configure the environment to use the Intel® Distribution of OpenVINO™ toolkit one time per session by running the following command:

    source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5

## Run the application

Start by changing the current directory to wherever you have git cloned the application code. For example:

    cd <path_to_the_store-aisle-python_directory>

### Sample Video

You can download sample video by running following commands:

    mkdir resources
    cd resources
    wget https://github.com/intel-iot-devkit/sample-videos/raw/master/store-aisle-detection.mp4
    cd .. 

### Running on the CPU

When running Intel® Distribution of OpenVINO™ toolkit Python applications on the CPU, the CPU extension library is required. This can be found at 

    /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/
    
To see a list of the various options that the application provides, execute the command:

    python3 main.py --help

Though, by default application runs on CPU, this can also be explicitly specified by `-d CPU` command-line argument:

    python3 main.py -m /opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_detection/pedestrian/rmnet_ssd/0013/dldt/person-detection-retail-0013.xml -l /opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.7  -i resources/store-aisle-detection.mp4

### Running on the GPU

To run on the integrated Intel® GPU in 32-bit mode, use the below command.

    python3 main.py -m /opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_detection/pedestrian/rmnet_ssd/0013/dldt/person-detection-retail-0013.xml -d GPU -pt 0.7  -i resources/store-aisle-detection.mp4 


To run on the integrated Intel® GPU in 16-bit mode, use the below command.

    python3 main.py -m /opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_detection/pedestrian/rmnet_ssd/0013/dldt/person-detection-retail-0013-fp16.xml -d GPU -pt 0.7  -i resources/store-aisle-detection.mp4

### Running on the Intel® Neural Compute Stick
To run on the Intel® Neural Compute Stick, use the ```-d MYRIAD``` command-line argument:

    python3 main.py -d MYRIAD -m /opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_detection/pedestrian/rmnet_ssd/0013/dldt/person-detection-retail-0013-fp16.xml -pt 0.7  -i resources/store-aisle-detection.mp4

**Note:** The Intel® Neural Compute Stick can only run FP16 models. The model that is passed to the application, through the `-m <path_to_model>` command-line argument, must be of data type FP16.

### Running on the HDDL
To run on the HDDL-R, use the `-d HETERO:HDDL,CPU` command-line argument:

    python3 main.py -m /opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_detection/pedestrian/rmnet_ssd/0013/dldt/person-detection-retail-0013-fp16.xml -l /opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_sse4.so -d HETERO:HDDL,CPU -i resources/store-aisle-detection.mp4

**Note:** The HDDL-R can only run FP16 models. The model that is passed to the application, through the `-m <path_to_model>` command-line argument, must be of data type FP16.

### Running on the FPGA

Before running the application on the FPGA,  program the AOCX (bitstream) file. Use the setup_env.sh script from [fpga_support_files.tgz](https://clicktime.symantec.com/38YrYPLxxSqYhBQLb1eruhW7Vc?u=http%3A%2F%2Fregistrationcenter-download.intel.com%2Fakdlm%2Firc_nas%2F12954%2Ffpga_support_files.tgz) to set the environment variables.<br>
For example:

    source /home/<user>/Downloads/fpga_support_files/setup_env.sh

The bitstreams for HDDL-F can be found under the `/opt/intel/openvino/bitstreams/a10_vision_design_bitstreams` folder.<br><br>To program the bitstream use the below command:<br>

    aocl program acl0 /opt/intel/openvino/bitstreams/a10_vision_design_bitstreams/2019R1_PL1_FP11_RMNet.aocx

For more information on programming the bitstreams, please refer to https://software.intel.com/en-us/articles/OpenVINO-Install-Linux-FPGA#inpage-nav-11

To run the application on the FPGA with floating point precision 16 (FP16), use the `-d HETERO:FPGA,CPU` command-line argument:

    python3 main.py -m /opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_detection/pedestrian/rmnet_ssd/0013/dldt/person-detection-retail-0013-fp16.xml -l /opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_sse4.so -d HETERO:FPGA,CPU -i resources/store-aisle-detection.mp4

### Using camera stream instead of video file

To get the input stream from the **camera**, use `-i cam` command-line argument. For example:


    python3 main.py -m /opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_detection/pedestrian/rmnet_ssd/0013/dldt/person-detection-retail-0013.xml -l /opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_sse4.so -i cam -d CPU -pt 0.7

## (Optional) Saving snapshots to the Cloud 
To upload the results to the cloud, the Microsoft Azure storage name and storage key are provided as the command line arguments.
Use `-an` and `-ak` options to specify Microsoft Azure storage name and storage key respectively.

    python3 main.py -m /opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_detection/pedestrian/rmnet_ssd/0013/dldt/person-detection-retail-0013.xml -l /opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.7  -i resources/store-aisle-detection.mp4 -an <azure-account-name> -ak <azure-account-key> 

**Note:** <br>
To obtain account name and account key from the Microsoft Azure portal, please refer:
https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python#copy-your-credentials-from-the-azure-portal

To view the uploaded snapshots on cloud, please refer:
https://docs.microsoft.com/en-us/azure/storage/blobs/storage-upload-process-images?tabs=net#verify-the-image-is-shown-in-the-storage-account

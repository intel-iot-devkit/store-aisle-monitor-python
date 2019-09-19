

# Store Aisle Monitor


| Details               |                  |
|-----------------------|------------------|
| Target OS             |  Ubuntu\* 16.04 LTS     |
| Programming Language  |  Python* 3.5 |
| Time to complete      |  30 min      |


![Output snapshot](./docs/images/figure2.png)

## Introduction

This reference implementation counts the number of people present in an image and generates a motion heatmap. It takes the input from the camera, or a video file for processing. Snapshots of the output are taken at regular intervals and are uploaded to the cloud. It also stores the snapshots of the output locally.

## Requirements

### Hardware
*  6th to 8th generation Intel® Core™ processors with Intel® Iris® Pro graphics or Intel® HD Graphics

### Software

* [Ubuntu* 16.04](http://releases.ubuntu.com/16.04/)
* OpenCL™ Runtime Package<br>
  **Note**: We recommend using a 4.14+ kernel to use this software. Run the following command to determine your kernel version:
  ```  
  uname -a
  ```  
* Intel® Distribution of OpenVINO™ toolkit 2019 R2 Release
* Microsoft Azure* Python SDK

## How it Works
- The application uses a video source, such as a camera or a video file, to grab the frames. The [OpenCV functions](https://docs.opencv.org/3.4/dd/d43/tutorial_py_video_display.html) are used to calculate frame width, frame height and frames per second (fps) of the video source. The application counts the number of people and generates motion heatmap.

![Architectural diagram](./docs/images/figure1.png)

-  People counter: A trained neural network model detects the people in the frame and bounding boxes are drawn on the people detected. This reference implementation uses a pre-trained model **person-detection-retail-0013** that can be downloaded using the **model downloader**, provided by the Intel® Distribution of OpenVINO™ toolkit.

- Motion Heatmap generation: An accumulated frame is used, on which every frame is added after preprocessing. This accumulated frame is used to generate the motion heatmap using [applyColorMap](https://docs.opencv.org/3.4/d3/d50/group__imgproc__colormap.html#gadf478a5e5ff49d8aa24e726ea6f65d15). The original frame and heatmap frame are merged using [addWeighted](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_core/py_image_arithmetics/py_image_arithmetics.html), to visualize the movement patterns over time.

-  The heatmap frame and people counter frame are merged using [addWeighted](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_core/py_image_arithmetics/py_image_arithmetics.html) and this merged frame is saved locally at regular intervals. The output is present in the *output_snapshots* directory of the project directory.

-  The application also uploads the results to the Microsoft Azure cloud at regular intervals, if a Microsoft Azure storage name and key are provided.

    ![Uploading snapshots to cloud](./docs/images/figure3.png)

## Setup

### Install Intel® Distribution of OpenVINO™ toolkit

Refer to [https://software.intel.com/en-us/articles/OpenVINO-Install-Linux](https://software.intel.com/en-us/articles/OpenVINO-Install-Linux) for more information on how to install and setup the Intel® Distribution of OpenVINO™ toolkit.

The OpenCL™ Runtime package is required to run the inference on a GPU. It is not mandatory for CPU inference.

### Other dependencies
**Microsoft Azure python SDK**<br>
The Azure python SDK allows you to build applications against Microsoft Azure Storage. [Azure Storage](https://docs.microsoft.com/en-us/azure/storage/common/storage-introduction) is Microsoft's cloud storage solution for modern data storage scenarios. Azure Storage offers a massively scalable object store for data objects, a file system service for the cloud, a messaging store for reliable messaging, and a NoSQL store.



### Which model to use

This application uses the [**person-detection-retail-0013**](https://docs.openvinotoolkit.org/2019_R1/_person_detection_retail_0013_description_person_detection_retail_0013.html) Intel® pre-trained model, that can be accessed using the **model downloader**. The **model downloader** downloads the __.xml__ and __.bin__ files that will be used by the application.

To download the model and install the dependencies of the application, run the below command in the `store-aisle-monitor-python` directory:
```
./setup.sh
```

### The Config File

The _resources/config.json_ contains the path of video that will be used by the application as input.

For example:
   ```
   {
       "inputs": [
          {
              "video":"path_to_video/video1.mp4"
          }
       ]
   }
   ```

The `path/to/video` is the path to an input video file.

### Which Input Video to use

We recommend using [store-aisle-detection](https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/store-aisle-detection.mp4).
For example:
   ```
   {
       "inputs": [
          {
              "video":"sample-videos/store-aisle-detection.mp4
          }
       ]
   }
   ```
If the user wants to use any other video, it can be used by providing the path in the config.json file.


### Using the Camera Stream instead of video

Replace `path/to/video` with the camera ID in the config.json file, where the ID is taken from the video device (the number X in /dev/videoX).

On Ubuntu, to list all available video devices use the following command:

```
ls /dev/video*
```

For example, if the output of above command is __/dev/video0__, then config.json would be:

```
  {
     "inputs": [
        {
           "video":"0"
        }
     ]
  }
```

### Setup the environment

Configure the environment to use the Intel® Distribution of OpenVINO™ toolkit one time per session by running the following command:

    source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5

**Note:** This command needs to be executed only once in the terminal where the application will be executed. If the terminal is closed, the command needs to be executed again.


### Run the application on Jupyter*

Go to the store-aisle-monitor-python directory and open the Jupyter notebook by running the following commands:

```
cd <path_to_the_store-aisle-monitor-python_directory>/Jupyter
jupyter notebook
```

#### Follow the steps to run the code on Jupyter*:

![Jupyter Notebook](./docs/images/jupy1.png)


1. Click on **New** button on the right side of the Jupyter window.

2. Click on **Python 3** option from the drop down list.

3. In the first cell type **import os** and press **Shift+Enter** from the keyboard.

4. Export the below environment variables in second cell of Jupyter and press **Shift+Enter**.
    ```
    %env MODEL = /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/Retail/object_detection/pedestrian/rmnet_ssd/0013/dldt/FP32/person-detection-retail-0013.xml
    %env CPU_EXTENSION = /opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_sse4.so 
    %env INPUT = resources/store-aisle-detection.mp4
    ```
5. User can set the threshold for the detection (PROB_THRESHOLD) and target device to infer on (DEVICE).
   Export these environment variables as given below if required, else skip this step. If user skips this step, these values are set to default values.
   ```
   %env DEVICE = CPU
   %env PROB_THRESHOLD = 0.7
   ```
 
6. To upload the results to the Microsoft Azure cloud (optional), export the below environment variables with an appropriate Microsoft Azure storage name and key.
   ```
   %env ACCOUNT_NAME = <enter-azure-account-name>
   %env ACCOUNT_KEY = <enter-azure-account-key>
   ```    
7. To run the application on sync mode, export the environment variable **%env FLAG = sync**. By default, the application runs on async mode.
8. Copy the code from **store_aisle_monitor_jupyter.py** and paste it in the next cell and press **Shift+Enter**.

9. Alternatively, code can be run in the following way:

    i. Click on the **store_aisle_monitor_jupyter.ipynb** notebook file from the Jupyter notebook home window.
    
    ii. Click on the **Kernel** menu and then select **Restart & Run All** from the drop down list.
    
    iii. On the pop-up window, click on **Restart and Run All Cells**.

    ![Jupyter Notebook](./docs/images/jupy2.png)

**NOTE:**

1. To run the application on **GPU**:
     * With the floating point precision 32 (FP32), change the **%env DEVICE = CPU** to **%env DEVICE = GPU**
     * With the floating point precision 16 (FP16), change the environment variables as given below:<br>

           %env DEVICE = GPU
           %env MODEL=/opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/Retail/object_detection/pedestrian/rmnet_ssd/0013/dldt/FP16/person-detection-retail-0013.xml 
     * **CPU_EXTENSION** environment variable is not required.
   
2. To run the application on **Intel® Neural Compute Stick**: 
      * Change the **%env DEVICE = CPU** to **%env DEVICE = MYRIAD**
      * The Intel® Neural Compute Stick can only run FP16 models. Hence change the environment variable for the model as shown below. <br>
              
            %env MODEL=/opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/Retail/object_detection/pedestrian/rmnet_ssd/0013/dldt/FP16/person-detection-retail-0013.xml
      * **CPU_EXTENSION** environment variable is not required.

3. To run the application on **Intel® Movidius™ VPU**:
     * Change the **%env DEVICE = CPU** to **%env DEVICE = HDDL**
     * The HDDL-R can only run FP16 models. Change the environment variable for the model as shown below  and the models that are passed to the application must be of data type FP16. <br>
 
           %env MODEL=/opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/Retail/object_detection/pedestrian/rmnet_ssd/0013/dldt/FP16/person-detection-retail-0013.xml
     * **CPU_EXTENSION** environment variable is not required.
<!--
4. To run the application on **FPGA**:
     * Change the **%env DEVICE = CPU** to **%env DEVICE = HETERO:FPGA,CPU**
     * With the **floating point precision 16 (FP16)**, change the path of the model in the environment variable **MODEL** as given below:<br>
      
           %env MODEL=/opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/Retail/object_detection/pedestrian/rmnet_ssd/0013/dldt/FP16/person-detection-retail-0013.xml
     * Export the **CPU_EXTENSION** environment variable as shown below:
         
           %env CPU_EXTENSION = /opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_sse4.so
-->

4. To obtain **account name** and **account key** from **azure portal**, please refer:
   https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python#copy-your-credentials-from-the-azure-portal
5. To view the uploaded snapshots on cloud, please refer:
   https://docs.microsoft.com/en-us/azure/storage/blobs/storage-upload-process-images?tabs=net#verify-the-image-is-shown-in-the-storage-account
   

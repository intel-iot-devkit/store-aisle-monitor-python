"""Store Aisle Monitor"""

"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
from argparse import ArgumentParser
import pathlib
import cv2
import numpy as np
from azure.storage.blob import BlockBlobService, PublicAccess
from inference import Network

# Weightage/ratio to merge (for Heatmap) original frame and colorMap frame(sum of both should be 1)
INITIAL_FRAME_WEIGHTAGE = 0.65
COLORMAP_FRAME_WEIGHTAGE = 0.35

# Weightage/ratio to merge (for integrated output) people count frame and colorMap frame(sum of both should be 1)
P_COUNT_FRAME_WEIGHTAGE = 0.65
COLORMAP_FRAME_WEIGHTAGE_1 = 0.35

# Multiplication factor to compute time interval for uploading snapshots to the cloud
MULTIPLICATION_FACTOR = 5

# Azure Blob container name
CONTAINER_NAME = 'store-aisle-monitor-snapshots'

# To get current working directory
CWD = os.getcwd()
# Creates subdirectories to save output videos and snapshots
pathlib.Path(CWD + '/output_snapshots/').mkdir(parents=True, exist_ok=True)


def apply_time_stamp_and_save(image, people_count, upload_azure):
    """
    Saves snapshots with timestamps.
    """
    current_date_time = time.strftime("%y-%m-%d_%H:%M:%S", time.gmtime())
    file_name = current_date_time + "_PCount_" + str(people_count) + ".png"
    file_path = CWD + "/output_snapshots/"
    local_file_name = "output_" + file_name
    file_name = file_path + local_file_name
    cv2.imwrite(file_name, image)
    if upload_azure is 1:
        upload_snapshot(file_path, local_file_name)


def create_cloud_container(account_name, account_key):
    """
    Creates a BlockBlobService container on cloud.
    """
    global BLOCK_BLOB_SERVICE

    # Create the BlockBlobService to call the Blob service for the storage account
    BLOCK_BLOB_SERVICE = BlockBlobService(account_name, account_key)
    # Create BlockBlobService container
    BLOCK_BLOB_SERVICE.create_container(CONTAINER_NAME)
    # Set the permission so that the blobs are public
    BLOCK_BLOB_SERVICE.set_container_acl(CONTAINER_NAME, public_access=PublicAccess.Container)


def upload_snapshot(file_path, local_file_name):
    """
    Uploads snapshots to cloud storage container.
    """
    try:

        full_path_to_file = file_path + local_file_name
        print("\nUploading to cloud storage as blob : " + local_file_name)
        # Upload the snapshot, with local_file_name as the blob name
        BLOCK_BLOB_SERVICE.create_blob_from_path(CONTAINER_NAME, local_file_name, full_path_to_file)

    except Exception as e:
        print(e)


def main():
    model_xml = (os.environ["MODEL"])
    input_source = (os.environ["INPUT"])
    device = os.environ['DEVICE'] if 'DEVICE' in os.environ.keys() else 'CPU'
    cpu_extension = os.environ[
        'CPU_EXTENSION'] if 'CPU_EXTENSION' in os.environ.keys() else None
    try:
        # Probability threshold for detections filtering
        prob_threshold = float(os.environ['PROB_THRESHOLD'])
    except KeyError:
        prob_threshold = 0.5
    try:
        # Specify the azure storage name to upload results to cloud.
        account_name = os.environ['ACCOUNT_NAME']
    except:
        account_name = None
    try:
        # Specify the azure storage key to upload results to cloud.
        account_key = os.environ['ACCOUNT_KEY']
    except:
        account_key = None                

    if account_name is "" or account_key is "":
        print("Invalid account name or account key!")
        sys.exit(1)
    elif account_name is not None and account_key is None:
        print("Please provide account key using -ak option!")
        sys.exit(1)        
    elif account_name is None and account_key is not None:
        print("Please provide account name using -an option!")
        sys.exit(1) 
    elif account_name is None and account_key is None:
        upload_azure = 0
    else:
        print("Uploading the results to Azure storage \""+ account_name+ "\"" )
        upload_azure = 1
        create_cloud_container(account_name, account_key)

    if input_source == 'cam':
        input_stream = 0
    else:
        input_stream = input_source
        assert os.path.isfile(input_source), "Specified input file doesn't exist"

    cap = cv2.VideoCapture(input_stream)
    if cap is None or not cap.isOpened():
        print('Warning: unable to open video source: ', input_source)
        sys.exit(1)

    # Initialise the class
    infer_network = Network()
    # Load the network to IE plugin to get shape of input layer
    n, c, h, w = infer_network.load_model(model_xml, device, 1, 1, 0, cpu_extension)[1]

    print("To stop the execution press Esc button")
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 1
    accumulated_image = np.zeros((initial_h, initial_w), np.uint8)
    mog = cv2.createBackgroundSubtractorMOG2()
    ret, frame = cap.read()
    while cap.isOpened():
        ret, next_frame = cap.read()
        if not ret:
            break
        frame_count = frame_count + 1
        in_frame = cv2.resize(next_frame, (w, h))
        # Change data layout from HWC to CHW
        in_frame = in_frame.transpose((2, 0, 1))  
        in_frame = in_frame.reshape((n, c, h, w))

        # Start asynchronous inference for specified request.
        inf_start = time.time()
        infer_network.exec_net(0, in_frame)
        # Wait for the result
        infer_network.wait(0)
        det_time = time.time() - inf_start
        people_count = 0

        # Converting to Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Remove the background
        fgbgmask = mog.apply(gray)

        # Thresholding the image
        thresh = 2
        max_value = 2
        threshold_image = cv2.threshold(fgbgmask, thresh, max_value,
                                                      cv2.THRESH_BINARY)[1]
        # Adding to the accumulated image
        accumulated_image = cv2.add(threshold_image, accumulated_image)
        colormap_image = cv2.applyColorMap(accumulated_image, cv2.COLORMAP_HOT)

        # Results of the output layer of the network
        res = infer_network.get_output(0)
        for obj in res[0][0]:
            # Draw only objects when probability more than specified threshold
            if obj[2] > prob_threshold:
                xmin = int(obj[3] * initial_w)
                ymin = int(obj[4] * initial_h)
                xmax = int(obj[5] * initial_w)
                ymax = int(obj[6] * initial_h)
                class_id = int(obj[1])
                # Draw bounding box
                color = (min(class_id * 12.5, 255), min(class_id * 7, 255),
                              min(class_id * 5, 255))
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                people_count = people_count + 1

        people_count_message = "People Count : " + str(people_count)
        inf_time_message = "Inference time: {:.3f} ms".format(det_time * 1000)
        cv2.putText(frame, inf_time_message, (15, 25), cv2.FONT_HERSHEY_COMPLEX, 1,
                         (255, 255, 255), 2)
        cv2.putText(frame, people_count_message, (15, 65), cv2.FONT_HERSHEY_COMPLEX, 1,
                         (255, 255, 255), 2)
        final_result_overlay = cv2.addWeighted(frame, P_COUNT_FRAME_WEIGHTAGE,
                                                    colormap_image,
                                                    COLORMAP_FRAME_WEIGHTAGE_1, 0)
        cv2.imshow("Detection Results", final_result_overlay)

        time_interval = MULTIPLICATION_FACTOR * fps
        if frame_count % time_interval == 0:
            apply_time_stamp_and_save(final_result_overlay, people_count, upload_azure)

        frame = next_frame

        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    infer_network.clean()


if __name__ == '__main__':
    sys.exit(main() or 0)




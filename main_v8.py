import cv2 as cv
import numpy as np
import os
import sys

# ANSI escape codes for colors
class TerminalColor:
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    END = '\033[0m'

# Initialize the parameters
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4   # Non-maximum suppression threshold
inpWidth = 608       # Width of network's input image
inpHeight = 608      # Height of network's input image

# Load command line arguments (replace these with actual paths for testing)
input_folder = sys.argv[1]  # Folder containing images to process
output_folder = sys.argv[2] # Folder where processed images will be saved
classesFile = sys.argv[3]
modelConfiguration = sys.argv[4]
modelWeights = sys.argv[5]

# Load names of classes and find the index for 'spike'
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
spike_index = classes.index('spike')  # Adjust this if 'spike' is not the correct name

# Load the network
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

# Check if CUDA is available and set preferable backend and target
if cv.cuda.getCudaEnabledDeviceCount() > 0:
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
else:
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Function to get the names of the output layers
# Get the names of the output layers
def getOutputsNames(net):
    layersNames = net.getLayerNames()
    outLayers = net.getUnconnectedOutLayers()
    if outLayers.ndim == 1:
        return [layersNames[i - 1] for i in outLayers]
    else:
        return [layersNames[i[0] - 1] for i in outLayers]

# Function to draw bounding boxes and increment spike counter
def drawPred(classId, conf, left, top, right, bottom, frame, count):
    if classId == spike_index:
        cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
        count += 1
    return count

# Process detections and draw bounding boxes
import numpy as np  # Ensure NumPy is imported

def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    spike_count = 0

    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    # Convert indices to a numpy array and flatten
    indices = np.array(indices).flatten()
    
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        spike_count = drawPred(classIds[i], confidences[i], left, top, left + width, top + height, frame, spike_count)

    return spike_count



# Ensure output folder exists
import pandas as pd  # Ensure pandas is imported

# Initialize a list to hold the image names and spike counts
results = []

# Ensure output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each image in the input folder
for file_name in os.listdir(input_folder):
    image_path = os.path.join(input_folder, file_name)
    if os.path.isfile(image_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        frame = cv.imread(image_path, cv.IMREAD_UNCHANGED)
        if frame.shape[2] == 4:
            frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)
        
        blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
        net.setInput(blob)
        outs = net.forward(getOutputsNames(net))
        
        spike_count = postprocess(frame, outs)
        
        # Append the file name and spike count to the results list
        results.append([file_name, spike_count])
        
        # Save the image with its original name, without adding any suffix
        output_file_path = os.path.join(output_folder, file_name)
        cv.imwrite(output_file_path, frame)
        
        print(TerminalColor.BLUE + f"Processed {file_name}: Found {spike_count} spikes." + TerminalColor.END)

# Convert the results list to a pandas DataFrame
df = pd.DataFrame(results, columns=['Image Name', 'Spike Count'])

# Define the Excel file path (you can choose a different name if you prefer)
excel_file_path = os.path.join(output_folder, 'Spike_Counts.xlsx')

# Save the DataFrame to an Excel file
df.to_excel(excel_file_path, index=False)

print(TerminalColor.GREEN + f"Spike counts have been saved to {excel_file_path}" + TerminalColor.END)

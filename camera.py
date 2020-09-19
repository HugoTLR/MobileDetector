#Built-in
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import argparse
import time

#3rd Party
from numpy import argpartition
from numpy import squeeze
from numpy import squeeze
from cv2 import cvtColor, COLOR_BGR2RGB
from cv2 import destroyAllWindows
from cv2 import FONT_HERSHEY_SIMPLEX
from cv2 import rectangle, resize
from cv2 import imshow, imwrite
from cv2 import putText
from cv2 import VideoCapture
from cv2 import waitKey
from tflite_runtime.interpreter import Interpreter
import glob


def load_labels(path):
  with open(path,'r') as f:
    #return {i: line.strip() for i,line in enumerate(f.readlines())}
    lines = f.readlines()
    labels = {}
    for row_number, content in enumerate(lines):
      pair = re.split(r'[:\s]+',content.strip(), maxsplit=1)
      if len(pair) == 2 and pair[0].strip().isdigit():
        labels[int(pair[0])] = pair[1].strip()
      else:
        labels[row_number] = pair[0].strip()
  return labels

def set_input_tensor(interpreter,image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image

def get_output_tensor(interpreter,index):
  output_details = interpreter.get_output_details()[index]
  tensor = squeeze(interpreter.get_tensor(output_details['index']))
  return tensor

def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  set_input_tensor(interpreter, image)
  interpreter.invoke() #From 30FPS to 2FPS lul, find a way to optimize that

  # Get all output details
  boxes = get_output_tensor(interpreter, 0)
  classes = get_output_tensor(interpreter, 1)
  scores = get_output_tensor(interpreter, 2)
  count = int(get_output_tensor(interpreter, 3))
  results = [{'bounding_box': boxes[i],'class_id':classes[i],'score': scores[i]} for i in range(count) if scores[i] >= threshold  ]
  return results

def annotate(frame, results, labels):
  """Draws the bounding box and label for each object in the results."""
  for obj in results:
    # Convert the bounding box figures from relative coordinates
    # to absolute coordinates based on the original resolution
    ymin, xmin, ymax, xmax = obj['bounding_box']
    xmin = int(xmin * CAMERA_WIDTH)
    xmax = int(xmax * CAMERA_WIDTH)
    ymin = int(ymin * CAMERA_HEIGHT)
    ymax = int(ymax * CAMERA_HEIGHT)
    # Overlay the box, label, and score on the camera preview
    rectangle(frame,(xmin,ymin),(xmax,ymax),(255,0,0))
    putText(frame,f"{labels[obj['class_id']+1]}, {obj['score']:.3f}",\
        (xmin,ymin+30),FONT_HERSHEY_SIMPLEX,1,(0,0,0),1)
    
def classify_image(interpreter, image, top_k=1):
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = squeeze(interpreter.get_tensor(output_details['index']))

  if output_details['dtype'] == uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

  ordered = argpartition(-output, top_k)
  return [(i,output[i]) for i in ordered[:top_k]]

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--model', help='File path of .tflite file.',\
     required=True)
  parser.add_argument('--labels', help='File path of labels file',\
     required=True)
  parser.add_argument('--threshold',\
             help='Score threshold for detected objects.',\
            required=False,\
            type=float,\
            default=0.6)
  args= parser.parse_args()
  
  threshold_value = args.threshold
  labels = load_labels(args.labels)
  interpreter = Interpreter(args.model)
  interpreter.allocate_tensors()
  _, height, width, _ = interpreter.get_input_details()[0]['shape']
  print(f"{height} {width}")

  cap = VideoCapture(0)
  CAMERA_WIDTH,CAMERA_HEIGHT = 640,480
  while True:
    start = time.time()
    ret, frame = cap.read()
    if not ret:
      break
    orig = resize(frame,(CAMERA_WIDTH,CAMERA_HEIGHT))
    frame = resize(cvtColor(orig,COLOR_BGR2RGB),(width,height))
    results = detect_objects(interpreter, frame, threshold_value)
    #annotate(orig,results,labels)

    imshow("Frame",orig)
    
    print(f'Avg FPS: {1/(time.time()-start)}')
    
    key = waitKey(1)
    if key  == ord('q') & 0xFF:
      break
    elif key == ord('s') & 0xFF:
      nb_imgs = len(glob.glob("./images/*.*"))
      imwrite(f"./images/{nb_imgs:03d}.jpg",orig)
  destroyAllWindows()
  cap.release()

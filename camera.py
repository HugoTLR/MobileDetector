#3rd Party
import numpy as np
import cv2 as cv
from tflite_runtime.interpreter import Interpreter
import glob


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import argparse
import time

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
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor

def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()

  # Get all output details
  boxes = get_output_tensor(interpreter, 0)
  classes = get_output_tensor(interpreter, 1)
  scores = get_output_tensor(interpreter, 2)
  count = int(get_output_tensor(interpreter, 3))

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
          'bounding_box': boxes[i],
          'class_id': classes[i],
          'score': scores[i]
      }
      results.append(result)
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
    #annotator.bounding_box([xmin, ymin, xmax, ymax])
    #annotator.text([xmin, ymin],
    #               '%s\n%.2f' % (labels[obj['class_id']], obj['score']))
    cv.rectangle(frame,(xmin,ymin),(xmax,ymax),(255,0,0))
    cv.putText(frame,f"{labels[obj['class_id']+1]}, {obj['score']:.3f}",\
        (xmin,ymin+30),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1)
def classify_image(interpreter, image, top_k=1):
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

  ordered = np.argpartition(-output, top_k)
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
  labels = load_labels(args.labels)
  interpreter = Interpreter(args.model)
  interpreter.allocate_tensors()
  _, height, width, _ = interpreter.get_input_details()[0]['shape']
  print(f"{height} {width}")

  cap = cv.VideoCapture(0)
  CAMERA_WIDTH,CAMERA_HEIGHT = 640,480
  while True:
    ret, frame = cap.read()
    if not ret:
      break
    orig = cv.resize(frame,(CAMERA_WIDTH,CAMERA_HEIGHT))
    frame = cv.resize(cv.cvtColor(frame,cv.COLOR_BGR2RGB),(width,height))
    #results = classify_image(interpreter,frame,1)
    results = detect_objects(interpreter, frame,args.threshold)
    annotate(orig,results,labels)
    #print(results)
    """
    pred = []
    for res in results:
      pred.append(str(labels[int(res[0])] + ':' + str(res[1])))
    print(pred)
    #print(f"{labels[int(results[0][0])]} : {str(results[0][1])}")
    """
    cv.imshow("Frame",orig)
    key = cv.waitKey(1)
    if key  == ord('q') & 0xFF:
      break
    elif key == ord('s') & 0xFF:
      nb_imgs = len(glob.glob("./images/*.*"))
      cv.imwrite(f"./images/{nb_imgs:03d}.jpg",frame)
  cv.destroyAllWindows()
  cap.release()

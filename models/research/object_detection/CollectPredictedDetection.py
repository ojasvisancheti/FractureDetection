def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

import io
import os
import scipy.misc
import pandas as pd
import numpy as np
import json
import six
import time
import glob
from IPython.display import display

from six import BytesIO
import cv2
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import tensorflow._api.v2.compat.v2 as tf
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util



def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: a file path (this can be local or on colossus)

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  # img_data = tf.io.gfile.GFile(path, 'rb').read()
  # image = Image.open(BytesIO(img_data))
  # (im_width, im_height) = image.size
  # return np.array(image.getdata()).reshape(
  #     (im_height, im_width, 3)).astype(np.uint8)

  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))

  out = image.convert("RGB")
  (im_width, im_height) = out.size
  # if channel > 1:
  #     gray= np.array(image.getdata()).reshape(
  #         (im_height, im_width, 1)).astype(np.uint8)
  #     return cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)

  return np.array(out.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

category_index = label_map_util.create_category_index_from_labelmap(r'object_detection\training\label_map.txt', use_display_name=True)

tf.keras.backend.clear_session()
model = tf.saved_model.load(r'D:\MSC\Sem2\AdaptixProject\FactureObjectDetection\models\research\object_detection\inference_graph\saved_model')


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


i = 1
coordinateList =list()
for image_path in glob.glob('D:/MSC/Sem2/AdaptixProject/FactureObjectDetection/Data/fracture/validation/*.jpg'):
  image_np = load_image_into_numpy_array(image_path)
  output_dict = run_inference_for_single_image(model, image_np)
  imagename = image_path
  coordinates = vis_util.return_coordinates(imagename,
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      use_normalized_coordinates=True,
      line_thickness=3,
      min_score_thresh=0.7)
  coordinateList = coordinateList + coordinates
column_name = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'score']
xml_df = pd.DataFrame(coordinateList, columns=column_name)
xml_df.to_csv(('object_detection/images/'+'predicted'+'_labels.csv'), index=None)


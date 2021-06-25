

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import cv2

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        list_with_all_boxes = []

        for box in root.iter('annotation'):
            for boxes in box.iter('object'):
                filename = path + "\\" + root.find('filename').text
                img = cv2.imread(filename)
                (rows, cols) = img.shape[:2]
                ymin = int(boxes.find("bndbox/ymin").text)
                xmin = int(boxes.find("bndbox/xmin").text)
                ymax = int(boxes.find("bndbox/ymax").text)
                xmax = int(boxes.find("bndbox/xmax").text)
                label = "fracture";
                value = (filename,
                         cols,
                         rows,
                         label,
                         xmin,
                         ymin,
                         xmax,
                         ymax,
                         )
                xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    for folder in ['train', 'test']:
        image_path = os.path.join(r'D:\MSC\Sem2\AdaptixProject\FactureObjectDetection\models\research\object_detection\images' + "\\", folder)
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv(('images/'+folder+'_labels.csv'), index=None)
    print('Successfully converted xml to csv.')


main()
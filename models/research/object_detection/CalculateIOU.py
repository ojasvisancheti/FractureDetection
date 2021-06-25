import pandas as pd
import ntpath


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA['xmin'], boxB['xmin'])
    yA = max(boxA['ymin'], boxB['ymin'])
    xB = min(boxA['xmax'], boxB['xmax'])
    yB = min(boxA['ymax'], boxB['ymax'])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA['xmax'] - boxA['xmin'] + 1) * (boxA['ymax'] - boxA['ymin'] + 1)
    boxBArea = (boxB['xmax'] - boxB['xmin'] + 1) * (boxB['ymax'] - boxB['ymin'] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


testcsv = pd.read_csv('object_detection/images/'+"test"+'_labels.csv')
predictedcsv = pd.read_csv('object_detection/images/'+'predicted'+'_labels.csv')
testcsv['filename'] = testcsv['filename'].apply(lambda x: ntpath.basename(x))
predictedcsv['filename'] = predictedcsv['filename'].apply(lambda x: ntpath.basename(x))

testcsv= testcsv.sort_values(['filename', 'xmax'], ascending=[True, False])
predictedcsv= predictedcsv.sort_values(['filename', 'xmax'], ascending=[True, False])
listIOU = list()
iteratedrrow = "NoRow"
i = 0
for row in predictedcsv.iterrows():
    if(iteratedrrow == str(row[1]["filename"])):
        iteratedrrow = row[1]["filename"]
        i = i + 1
    else:
        iteratedrrow = row[1]["filename"]
        i = 0

    actualOutput = testcsv[testcsv["filename"] == row[1]["filename"]]
    iou = bb_intersection_over_union(actualOutput.iloc[i], row[1])
    listIOU.append(iou)
print(sum(listIOU) / len(listIOU))



#0.8718566047961086

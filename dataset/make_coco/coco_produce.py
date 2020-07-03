import pandas as pd
import numpy as np
from .tools import make_json


def class_id_map(map_data):
    D = {}
    for i in range(len(map_data)):
        D.update({map_data[i][0]: map_data[i][1]})
    return D


if __name__ == '__main__':
    phase = "train"
    image_root = "/home/wbw/PAN/EPIC_KITCHES_2018/object_detection_images/" + phase + "2017/"
    root_box = "EPIC_train_object_labels.csv"
    root_class = "EPIC_noun_classes.csv"
    data_inf = pd.read_csv(root_box, dtype={"frame": np.str})
    data = data_inf[["noun_class", "noun", "participant_id", "video_id", "frame", "bounding_boxes"]]
    data = data.values

    data_inf2 = pd.read_csv(root_class)
    data2 = data_inf2[["noun_id", "class_key", "nouns"]]
    data2 = data2.values
    D = class_id_map(data2)

    total = {}
    count = 0
    for i in range(len(data)):
        if data[i][5] != "[]":
            noun_class = data[i][0]
            noun = data[i][1]
            image_name = data[i][2] + "/" + data[i][3] + "/0000" + str(data[i][4]) + ".jpg"
            box = list(map(int, data[i][5][1:-1].replace(" ", "").replace("(", "").replace(")", "").split(",")))
            count += 1
            if image_name not in total:
                total.update({image_name: [[noun_class, box]]})
            else:
                total[image_name].append([noun_class, box])
    make_json(total, D, phase, image_root)


import cv2
import json
import os


def make_json(data, class_index, phase, root_all_image):
    dataset = {"info": {"year": 2020, "version": "2020", "description": "for_kitchen", "contributor": "wbw", "url": "", "date_created": "2020.06.09"},
               "license": {"id": 1, "url": "", "name": "wangbowen"},
               "images": [],
               "annotations": [],
               "categories": []}
    total_bbox_id = 0
    for s, k in enumerate(list(class_index.keys())):
        if int(k) == 0:
            continue
        print(k)
        dataset["categories"].append({"id": k, "name": class_index[k]})

    for i, key in enumerate(data.keys()):
        if i % 100 == 0:
            print(str(i) + "/" + str(len(data)))
        if not os.path.exists(root_all_image+key):
            print(root_all_image+key)
            continue
        current_img = cv2.imread(root_all_image+key)
        height, weight, c = current_img.shape
        dataset["images"].append({"license": 1,
                                  "file_name": key,
                                  "id": i,
                                  "weight": weight,
                                  "height": height})
        box = data[key]
        for j in range(len(box)):
            dataset["annotations"].append({"area": box[j][1][0] * box[j][1][1],
                                           "bbox": [box[j][1][1], box[j][1][0], box[j][1][3], box[j][1][2]],
                                           "category_id": box[j][0],
                                           "id": total_bbox_id,
                                           "image_id": i,
                                           "iscrowd": 0,
                                           "segmentation": [[]]})
            total_bbox_id += 1

    with open("instances_" + phase + "2017.json", "w") as f:
        json.dump(dataset, f)
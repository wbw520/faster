from pycocotools.coco import COCO


coco_root = "/home/wbw/PAN/final_cxr/annotations/instances_train2017.json"
coco = COCO(coco_root)
img_id = list(sorted(coco.imgs.keys()))
print(img_id)
ann_ids = coco.getAnnIds(imgIds=img_id[0])
path = coco.loadImgs(img_id[0])[0]['file_name']
print(path)
print(ann_ids)
target = coco.loadAnns(ann_ids)
print(target)
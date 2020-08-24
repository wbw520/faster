from train import get_args_parser
import argparse
import torch
import torchvision.transforms as T
from lib.faster_rcnn import fasterrcnn_resnet_fpn
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
import pandas as pd
import os
from tools.tools import read_coco, cal_overlap, Filter
from tools.make_json import make_json


torch.set_grad_enabled(False)

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.ToTensor(),
])


def detect(im, model, transform):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)
    img = img.to(args.device)
    # propagate through the model
    outputs = model(img)[0]

    return outputs["boxes"], outputs["scores"], outputs["labels"]


def get_name(root, mode_folder=True):
    for root, dirs, file in os.walk(root):
        if mode_folder:
            return dirs
        else:
            return file


def plot_results(name, pil_img, boxes, labels, scores, CLASSES):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, cl, (xmin, ymin, xmax, ymax), c in zip(scores, labels, boxes.tolist(), COLORS * 100):
        if p > 0:
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, color="red", linewidth=3))
            # print(CLASSES[int(cl.cpu().detach().numpy())])
            text = f'{CLASSES[int(cl.cpu().detach().numpy())]}: {p:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.savefig("save_imgs/" + name + ".png")
    plt.show()
    plt.close()


def plot_add(name, pil_img, pre_boxes, true_boxes):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for (xmin, ymin, xmax, ymax) in pre_boxes:
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color="red", linewidth=3))
    for (xmin, ymin, xmax, ymax) in true_boxes:
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color="green", linewidth=3))
    plt.axis('off')
    plt.savefig("save_imgs/" + name + ".png")
    # plt.show()
    plt.close()


def class_id_map(map_data):
    D = {}
    for i in range(len(map_data)):
        D.update({map_data[i][0]: map_data[i][1]})
    return D


@torch.no_grad()
def main():
    # model = fasterrcnn_resnet_fpn(num_classes=2, pretrained=False)
    # model.to(args.device)
    # print("load model start:")
    # checkpoint = torch.load("saved_model_att/checkpoint0005.pth", map_location=args.device)
    # model.load_state_dict(checkpoint['model'])
    # model.eval()
    test = True
    vis = False
    if test:
        add = "for_test/"
    else:
        add = "nodule/"
    root = "/home/wbw/PAN/final_cxr/"+add

    record_pre = {}
    record_true = read_coco("/home/wbw/PycharmProjects/faster/instances_test.json", test=True)
    record_last = read_coco("/home/wbw/PAN/final_cxr/annotations/att.json")
    record_att = read_coco("/home/wbw/PycharmProjects/faster/att_test_175.json")

    # for v, k in record_att.items():
    #     name = v
    #     if v not in record_true:
    #         continue
    #     boxes = record_att[v]
    #     box_true = record_true[v]
    #     img = Image.open(root + name + ".png")
    #     plot_add(name, img, boxes, box_true)

    # if test:
    #     folders = get_name(root, mode_folder=False)
    # else:
    #     folders = []
    #     for v, k in record_last.items():
    #         folders.append(v)
    # for i, name in enumerate(tqdm(folders)):
    #     if test:
    #         img = Image.open(root + name)
    #     else:
    #         img = Image.open(root + name + "/total.png")
    #     boxes, scores, label = detect(img, model, transform)
    #     boxes = boxes.tolist()
    #     if vis:
    #         D = {1: "nodule"}
    #         # plot_results(name, img, boxes, label, scores, D)
    #         if name[:-4] not in record_true:
    #             continue
    #         box_true = record_true[name[:-4]]
    #         plot_add(name, img, boxes, box_true)
    #     if test:
    #         record_pre.update({name[:-4]: boxes})
    #     else:
    #         record_pre.update({name: boxes})

    if test:
        cal_overlap(record_att, record_true, thresh=0.1, mode="man")
    else:
        Filter().factory(record_last, record_pre, "second")
        # make_json(record_pre, phase="first", root_all_image="/home/wbw/PAN/final_cxr/nodule")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main()
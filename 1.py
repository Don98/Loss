import numpy as np
from pycocotools.coco import COCO
import skimage.transform

def coco_label_to_label(coco_label):
    return coco_labels_inverse[coco_label]

coco = COCO("./instances_val2017.json")
# print(coco.getCatIds())
categories = coco.loadCats(coco.getCatIds())
image_ids = coco.getImgIds()
# print(categories)

categories.sort(key=lambda x: x['id'])
# print(categories[0])

classes             = {}
coco_labels         = {}
coco_labels_inverse = {}
for c in categories:
    # print(c)
    coco_labels[len(classes)] = c['id']
    coco_labels_inverse[c['id']] = len(classes)
    classes[c['name']] = len(classes)
print(coco_labels)
print(coco_labels_inverse)
print(classes)
labels = {}
for key, value in classes.items():
    labels[value] = key
# print(labels)
# print(image_ids[0])
annotations_ids = coco.getAnnIds(imgIds=image_ids[0], iscrowd=False)
# print(annotations_ids)
annotations     = np.zeros((0, 5))


# parse annotations
coco_annotations = coco.loadAnns(annotations_ids)
print(coco_annotations[0])

exit()
for idx, a in enumerate(coco_annotations):
    # print(idx,a)
    # some annotations have basically no width / height, skip them
    if a['bbox'][2] < 1 or a['bbox'][3] < 1:
        continue

    annotation        = np.zeros((1, 5))
    annotation[0, :4] = a['bbox']
    annotation[0, 4]  = coco_label_to_label(a['category_id'])
    # print(a['category_id'])
    # print(annotation[0, 4])
    annotations       = np.append(annotations, annotation, axis=0)
    # exit()

# transform from [x, y, w, h] to [x1, y1, x2, y2]
annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
annotations[:, 3] = annotations[:, 1] + annotations[:, 3]
# print(annotations)

image_info = coco.loadImgs(image_ids[0])[0]
# print(image_info)

def resizer(min_side=608, max_side=1024):
    center = [460,640]
    # image, annots = sample['img'], sample['annot']

    annot = annotations
    rows, cols, cns = 1000,2000,3

    smallest_side = min(rows, cols)
    # print(smallest_side)
    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)

    if largest_side * scale > max_side:
        scale = max_side / largest_side

    # resize the image with the computed scale
    image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
    rows, cols, cns = image.shape

    pad_w = 32 - rows%32
    pad_h = 32 - cols%32

    new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
    new_image[:rows, :cols, :] = image.astype(np.float32)

    annots[:, :4] *= scale

    return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}


# img = load_image(0)
annot = annotations
# print(annot[:,:4])
# resizer()
coco_true = coco
# coco_pred = coco_true.loadRes('{}_bbox_results.json'.format("2017"))
# print(coco_true)
for i in coco_true:
    print(i)
# print(coco_pred)
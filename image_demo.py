from generators.pascal import PascalVocGenerator
from models.resnet import centernet
import cv2
import os
import numpy as np
import time
from generators.utils import affine_transform, get_affine_transform
import os.path as osp



def preprocess_image(image, c, s, tgt_w, tgt_h):
    trans_input = get_affine_transform(c, s, (tgt_w, tgt_h))
    image = cv2.warpAffine(image, trans_input, (tgt_w, tgt_h), flags=cv2.INTER_LINEAR)
    image = image.astype(np.float32)

    image[..., 0] -= 103.939
    image[..., 1] -= 116.779
    image[..., 2] -= 123.68

    return image

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


voc_classes = {
    'aeroplane': 0,
    'bicycle': 1,
    'bird': 2,
    'boat': 3,
    'bottle': 4,
    'bus': 5,
    'car': 6,
    'cat': 7,
    'chair': 8,
    'cow': 9,
    'diningtable': 10,
    'dog': 11,
    'horse': 12,
    'motorbike': 13,
    'person': 14,
    'pottedplant': 15,
    'sheep': 16,
    'sofa': 17,
    'train': 18,
    'tvmonitor': 19
}

model_path = '/home/tienduchoang/Videos/models/keras-centernet/checkpoints/pretrain_model.h5'
num_classes = len(voc_classes)
classes = list(voc_classes.keys())
flip_test = True
nms = True
keep_resolution = False
score_threshold = 0.1
colors = [np.random.randint(0, 256, 3).tolist() for i in range(num_classes)]
model, prediction_model, debug_model = centernet(num_classes=num_classes,
                                                 nms=nms,
                                                 flip_test=flip_test,
                                                 freeze_bn=True,
                                                 score_threshold=score_threshold)
prediction_model.load_weights(model_path, by_name=True, skip_mismatch=True)

# image = generator.load_image(i)
image_path = "./data/1.jpg"
image = cv2.imread(image_path)

src_image = image.copy()

c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
s = max(image.shape[0], image.shape[1]) * 1.0

input_size = 512
tgt_w = input_size
tgt_h = input_size
image = preprocess_image(image, c, s, tgt_w=tgt_w, tgt_h=tgt_h)
if flip_test:
    flipped_image = image[:, ::-1]
    inputs = np.stack([image, flipped_image], axis=0)
else:
    inputs = np.expand_dims(image, axis=0)
# run network
start = time.time()
detections = prediction_model.predict_on_batch(inputs)[0]
print(time.time() - start)
scores = detections[:, 4]
# select indices which have a score above the threshold
indices = np.where(scores > score_threshold)[0]

# select those detections
detections = detections[indices]
detections_copy = detections.copy()
detections = detections.astype(np.float64)
trans = get_affine_transform(c, s, (tgt_w // 4, tgt_h // 4), inv=1)

for j in range(detections.shape[0]):
    detections[j, 0:2] = affine_transform(detections[j, 0:2], trans)
    detections[j, 2:4] = affine_transform(detections[j, 2:4], trans)

detections[:, [0, 2]] = np.clip(detections[:, [0, 2]], 0, src_image.shape[1])
detections[:, [1, 3]] = np.clip(detections[:, [1, 3]], 0, src_image.shape[0])
for detection in detections:
    xmin = int(round(detection[0]))
    ymin = int(round(detection[1]))
    xmax = int(round(detection[2]))
    ymax = int(round(detection[3]))
    score = '{:.4f}'.format(detection[4])
    class_id = int(detection[5])
    color = colors[class_id]
    class_name = classes[class_id]
    label = '-'.join([class_name, score])
    ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(src_image, (xmin, ymin), (xmax, ymax), color, 1)
    cv2.rectangle(src_image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
    cv2.putText(src_image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imwrite('test/result.jpg', src_image)

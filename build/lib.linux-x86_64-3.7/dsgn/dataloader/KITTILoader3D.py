import numpy as np

from dsgn.dataloader.kitti_dataset import kitti_dataset as kittidataset

def get_kitti_annos(labels,
    # ignore_van_and_personsitting=False,
    # ignore_smaller=True,
    # ignore_occlusion=True,
    ignore_van_and_personsitting=False,
    ignore_smaller=False,
    ignore_occlusion=False,
    ignore_truncation=True,
    valid_classes=[1,2,3,4]):

    assert not ignore_occlusion # 6150 occlusion should be induced

    boxes = []
    box3ds = []
    ori_classes = []
    for i, label in enumerate(labels):
        # 4 will be ignored.
        if label.type == 'Pedestrian' or label.type == 'Person_sitting': typ = 1
        elif label.type == 'Car' or label.type == 'Van': typ = 2
        elif label.type == 'Cyclist': typ = 3
        elif label.type == 'DontCare': typ = 4
        elif label.type in ['Misc', 'Tram', 'Truck']: continue
        else:
            raise ValueError('Invalid Label.')

        # only train Car or Person
        if typ != 4 and typ not in set(valid_classes) - set([4]):
            continue

        if ignore_van_and_personsitting and (label.type == 'Van' or label.type == 'Person_sitting'):
            typ = 4

        if ignore_smaller and label.box2d[3] - label.box2d[1] <= 10.:
            typ = 4

        if ignore_occlusion and label.occlusion >= 3:
            typ = 4

        if ignore_truncation and label.truncation >= 0.98:
            typ = 4

        if typ not in valid_classes:
            continue 

        boxes.append( np.array(label.box2d) )
        box3ds.append( np.array(label.box3d[[3,4,5, 0,1,2, 6]]) )
        ori_classes.append( typ )

        boxes[-1][2:4] = boxes[-1][2:4] - boxes[-1][0:2]

        # if typ == 4:
        #     box3ds[-1] = np.zeros((7,))

    boxes = np.asarray(boxes, dtype=np.float32)
    box3ds = np.asarray(box3ds, dtype=np.float32)
    ori_classes = np.asarray(ori_classes, dtype=np.int32)

    # inds = ori_classes.argsort()
    # boxes = boxes[inds]
    # box3ds = box3ds[inds]
    # ori_classes = ori_classes[inds]

    return boxes, box3ds, ori_classes

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath, train_file, depth_disp=False, cfg=None, is_train=False, generate_target=False):
    kitti_dataset = kittidataset('trainval').train_dataset

    left_fold = 'image_2/'
    right_fold = 'image_3/'
    if depth_disp:
        disp_L = 'depth/'
    else:
        disp_L = 'disparity/'

    with open(train_file, 'r') as f:
        train_idx = [x.strip() for x in f.readlines()]

    if is_train or generate_target:
        filter_idx = []
        if cfg.RPN3D_ENABLE:
            for image_index in train_idx:
                labels = kitti_dataset.get_label_objects(int(image_index))
                boxes, box3ds, ori_classes = get_kitti_annos(labels,
                    valid_classes = cfg.valid_classes)
                if len(box3ds) > 0:
                    filter_idx.append(image_index)
            train_idx = filter_idx

    left_train = [filepath + '/' + left_fold + img + '.png' for img in train_idx]
    right_train = [filepath + '/' + right_fold + img + '.png' for img in train_idx]
    disp_train_L = [filepath + '/' + disp_L + img + '.npy' for img in train_idx]

    return left_train, right_train, disp_train_L

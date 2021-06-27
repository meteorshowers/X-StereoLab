import time
import fire
import kitti_common as kitti
from eval import get_official_eval_result, get_coco_eval_result


def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def evaluate(label_path,
             result_path,
             current_class=0,
             coco=False,
             score_thresh=-1,
             eval_dist=None):
    dt_annos, image_ids = kitti.get_label_annos(result_path, return_image_ids=True, eval_dist=eval_dist)
    print('Eval {} images'.format(len(dt_annos)))
    if score_thresh > 0:
        dt_annos = kitti.filter_annos_low_score(dt_annos, score_thresh)
    #val_image_ids = _read_imageset_file(label_split_file)
    gt_annos = kitti.get_label_annos(label_path, image_ids, eval_dist=eval_dist)
    if coco:
        print(get_coco_eval_result(gt_annos, dt_annos, current_class))
    else:
        print(get_official_eval_result(gt_annos, dt_annos, current_class))


if __name__ == '__main__':
    fire.Fire()

import torch
import torch.nn.functional as F

from dsgn.utils.bounding_box import BoxList, compute_corners
from dsgn.utils.boxlist_ops import cat_boxlist
from dsgn.utils.boxlist_ops import remove_small_boxes

from dsgn.utils.torch_utils import *
from dsgn.utils.numpy_utils import *

import numpy as np

class FCOS3DPostProcessor(torch.nn.Module):
    def __init__(
        self,
        pre_nms_thresh,
        pre_nms_top_n,
        nms_thresh,
        fpn_post_nms_top_n,
        min_size,
        num_classes,
        cfg
    ):
        super(FCOS3DPostProcessor, self).__init__()
        self.cfg = cfg
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes
        self.anchors_y = torch.as_tensor(self.cfg.RPN3D.ANCHORS_Y)
        self.num_angles = cfg.num_angles
        self.anchor_angles = torch.as_tensor(self.cfg.ANCHOR_ANGLES)
        self.centerness4class = getattr(self.cfg, 'centerness4class', False)
        self.class4angles = getattr(self.cfg, 'class4angles', True)
        self.box_corner_parameters = getattr(self.cfg, 'box_corner_parameters', True)
        self.mul_centerness = getattr(self.cfg, 'SCORE_MUL_CENTERNESS', True)

        self.pred_reg_dim = 24 if self.box_corner_parameters else 7

        self.CV_X_MIN, self.CV_Y_MIN, self.CV_Z_MIN = cfg.CV_X_MIN, cfg.CV_Y_MIN, cfg.CV_Z_MIN
        self.CV_X_MAX, self.CV_Y_MAX, self.CV_Z_MAX = cfg.CV_X_MAX, cfg.CV_Y_MAX, cfg.CV_Z_MAX
        self.X_MIN, self.Y_MIN, self.Z_MIN = cfg.X_MIN, cfg.Y_MIN, cfg.Z_MIN
        self.X_MAX, self.Y_MAX, self.Z_MAX = cfg.X_MAX, cfg.Y_MAX, cfg.Z_MAX
        self.VOXEL_X_SIZE, self.VOXEL_Y_SIZE, self.VOXEL_Z_SIZE = cfg.VOXEL_X_SIZE, cfg.VOXEL_Y_SIZE, cfg.VOXEL_Z_SIZE

        if isinstance(self.cfg.RPN3D.NMS_THRESH, float):
            self.nms_thresh = [self.cfg.RPN3D.NMS_THRESH] * self.num_classes
        else:
            self.nms_thresh = self.cfg.RPN3D.NMS_THRESH

    def forward_for_single_feature_map(
            self, locations, box_cls,
            box_regression, centerness,
            image_sizes, targets=None,
            calibs_Proj=None):
        N, C, H, W = box_cls.shape

        if not self.class4angles:
            box_cls = box_cls.reshape(N, 1, self.num_classes, H, W).repeat(1, self.num_angles, 1, 1, 1)
        box_cls = box_cls.view(N, self.num_angles * self.num_classes, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, self.num_angles, self.num_classes).sigmoid() # sigmoid !!!
        box_regression = box_regression.view(N, self.num_angles * self.num_classes * self.pred_reg_dim, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, self.num_angles, self.num_classes, self.pred_reg_dim)
        if not self.centerness4class:
            centerness = centerness.view(N, self.num_angles, H, W).permute(0, 2, 3, 1)
            centerness = centerness[:, :, :, :, None].repeat(1, 1, 1, 1, self.num_classes)
        else:
            centerness = centerness.view(N, self.num_angles * self.num_classes, H, W).permute(0, 2, 3, 1)
            centerness = centerness.reshape(N, H, W, self.num_angles, self.num_classes)
        centerness = centerness.reshape(N, -1, self.num_angles, self.num_classes).sigmoid()

        candidate_inds = box_cls > self.pre_nms_thresh

        #hack at least one
        for i, j in enumerate(candidate_inds.reshape(N, -1).sum(1)):
            if j == 0:
                candidate_inds[i, 0, 0] = 1

        pre_nms_top_n = candidate_inds.reshape(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        # multiply the classification scores with centerness scores
        if self.mul_centerness:
            box_cls = box_cls * centerness

        results = []
        for i in range(N):
            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]

            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_angle = per_candidate_nonzeros[:, 1]
            per_class = per_candidate_nonzeros[:, 2]

            per_box_regression = box_regression[i]

            # hack it should be perclass - 1, but training use perclass
            per_box_regression = per_box_regression[per_box_loc, per_angle, per_class]
            per_locations = locations[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]
                per_class = per_class[top_k_indices]

            ys = self.anchors_y.cuda()[per_class]
            per_locations3d = torch.stack([per_locations[:, 0], ys, per_locations[:, 1]], dim=1)
            if self.box_corner_parameters:
                detection_3d = per_box_regression.reshape(-1, 8, 3) + per_locations3d[:, None]
            else:
                theta = per_box_regression[:, 6]
                detection_3d = compute_corners(per_box_regression[:, 3:6], theta).transpose(1,2)
                detection_3d = detection_3d + (per_locations3d + per_box_regression[:, :3])[:, None]
                detection_3d[:, :, 1] = detection_3d[:, :, 1] + per_box_regression[:, None, 3] / 2.

            detections = project_rect_to_image(detection_3d.reshape(-1, 3), calibs_Proj[i].float().cuda()).reshape(-1,8,2)
            detections = torch.cat([torch.min(detections, dim=1)[0], torch.max(detections, dim=1)[0]], dim=1)

            h, w = image_sizes[i]
            boxlist = BoxList(detections, (int(w), int(h)), mode="xyxy")
            boxlist.add_field("labels", per_class + 1)
            boxlist.add_field("scores", per_box_cls)
            boxlist.add_field('box_corner3d', detection_3d)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)

        return results

    def forward(self, box_cls, box_regression, centerness,
        image_sizes=None, targets=None, calibs_Proj=None):
        ### compute anchors & locaitons_bev
        locations_bev = compute_locations_bev(self.Z_MIN, self.Z_MAX, self.VOXEL_Z_SIZE, 
            self.X_MIN, self.X_MAX, self.VOXEL_X_SIZE, box_cls.device)

        boxlists = []
        sampled_boxes = self.forward_for_single_feature_map(
            locations_bev, box_cls, box_regression, centerness, image_sizes, 
            targets=targets, calibs_Proj=calibs_Proj
        )
        boxlists.append(self.select_over_all_levels(sampled_boxes))

        return boxlists

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            scores = boxlists[i].get_field("scores")
            labels = boxlists[i].get_field("labels")
            boxes = boxlists[i].bbox
            if boxlists[i].has_field('box_corner3d'):
                box_corner3d = boxlists[i].get_field('box_corner3d')
            boxlist = boxlists[i]
            result = []
            # skip the background
            for j in range(1, self.num_classes + 1):
                inds = (labels == j).nonzero().view(-1)

                scores_j = scores[inds]
                boxes_j = boxes[inds, :].view(-1, 4)
                boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
                boxlist_for_class.add_field("scores", scores_j)
                if boxlists[i].has_field('box_corner3d'):
                    box_corner3d_j = box_corner3d[inds]
                    boxlist_for_class.add_field('box_corner3d', box_corner3d_j)

                if len(inds) > 0:
                    bev_boxes = []
                    box_corner3d_j = boxlist_for_class.get_field('box_corner3d').cpu()
                    for k in range(len(box_corner3d_j)):
                        box_cor3d = box_corner3d_j[k].clone()
                        box_center3d = box_cor3d.mean(dim=0)
                        x, y, z = box_center3d
                        box_cor3d = box_cor3d - box_center3d[None,:]
                        h, w, l, r = get_dimensions(box_cor3d.transpose(0,1))
                        bev_boxes.append([x, z, w, l, r])
                    bev_boxes = np.asarray(bev_boxes)

                    from dsgn.utils.rotate_iou import compute_iou_fast
                    from dsgn.utils.nms import nms_givenIoU
                    iou = compute_iou_fast(bev_boxes, bev_boxes)
                    keep = nms_givenIoU(boxlist_for_class.get_field('scores').cpu().numpy(), self.nms_thresh[j-1], iou=iou)
                    boxlist_for_class = boxlist_for_class[keep]

                num_labels = len(boxlist_for_class)
                boxlist_for_class.add_field(
                    "labels", torch.full((num_labels,), j,
                                         dtype=torch.int64,
                                         device=scores.device)
                )
                result.append(boxlist_for_class)

            result = cat_boxlist(result)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.get_field("scores")
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                image_thresh = torch.clamp(image_thresh, min=0.05)
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            else:
                cls_scores = result.get_field("scores")
                keep = cls_scores >= 0.05
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]

            results.append(result)
        return results


def make_fcos3d_postprocessor(config):
    pre_nms_thresh = config.RPN3D.PRE_NMS_THRESH
    pre_nms_top_n = config.RPN3D.PRE_NMS_TOP_N
    nms_thresh = config.RPN3D.NMS_THRESH
    fpn_post_nms_top_n = config.RPN3D.POST_NMS_TOP_N

    box_selector = FCOS3DPostProcessor(
        pre_nms_thresh=pre_nms_thresh,
        pre_nms_top_n=pre_nms_top_n,
        nms_thresh=nms_thresh,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        min_size=0,
        num_classes=config.num_classes,
        cfg=config,
    )

    return box_selector

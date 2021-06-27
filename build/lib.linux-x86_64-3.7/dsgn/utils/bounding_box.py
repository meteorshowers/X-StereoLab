# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

import numpy as np

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class BoxList(object):
    """
    This class represents a set of bounding boxes.
    The bounding boxes are represented as a Nx4 Tensor.
    In order to uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as
    labels.
    """

    def __init__(self, bbox, image_size, mode="xyxy", ratios=(1., 1.), img_id=None):
        device = bbox.device if isinstance(bbox, torch.Tensor) else torch.device("cpu")
        bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)
        if bbox.ndimension() != 2:
            raise ValueError(
                "bbox should have 2 dimensions, got {}".format(bbox.ndimension())
            )
        if bbox.size(-1) != 4:
            raise ValueError(
                "last dimension of bbox should have a "
                "size of 4, got {}".format(bbox.size(-1))
            )
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")

        self.bbox = bbox
        self.size = image_size  # (image_width, image_height)
        self.img_id = img_id
        self.mode = mode
        self.extra_fields = {}
        self.ratios = ratios

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, bbox):
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v

    def convert(self, mode):
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")
        if mode == self.mode:
            return self
        # we only have two modes, so don't need to check
        # self.mode
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if mode == "xyxy":
            bbox = torch.cat((xmin, ymin, xmax, ymax), dim=-1)
            bbox = BoxList(bbox, self.size, mode=mode, ratios=self.ratios, img_id=self.img_id)
        else:
            TO_REMOVE = 1
            bbox = torch.cat(
                (xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE), dim=-1
            )
            bbox = BoxList(bbox, self.size, mode=mode, ratios=self.ratios, img_id=self.img_id)
        bbox._copy_extra_fields(self)
        return bbox

    def _split_into_xyxy(self):
        if self.mode == "xyxy":
            xmin, ymin, xmax, ymax = self.bbox.split(1, dim=-1)
            return xmin, ymin, xmax, ymax
        elif self.mode == "xywh":
            TO_REMOVE = 1
            xmin, ymin, w, h = self.bbox.split(1, dim=-1)
            return (
                xmin,
                ymin,
                xmin + (w - TO_REMOVE).clamp(min=0),
                ymin + (h - TO_REMOVE).clamp(min=0),
            )
        else:
            raise RuntimeError("Should not be here")

    def pad_rb(self, delta_w, delta_h):
        if delta_w > 0 or delta_h > 0:
            self.add_field('real_img_size', self.size)
            self.size = (self.size[0] + delta_w, self.size[1] + delta_h)
        return self

    def resize(self, size, *args, **kwargs):
        """
        Returns a resized copy of this bounding box

        :param size: The requested size in pixels, as a 2-tuple:
            (width, height).
        """

        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        self.ratios = ratios
        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            scaled_box = self.bbox * ratio
            bbox = BoxList(scaled_box, size, mode=self.mode, ratios=self.ratios, img_id=self.img_id)
            # bbox._copy_extra_fields(self)
            for k, v in self.extra_fields.items():
                if not isinstance(v, torch.Tensor):
                    v = v.resize(size, *args, **kwargs)
                bbox.add_field(k, v)
            return bbox

        ratio_width, ratio_height = ratios
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        scaled_xmin = xmin * ratio_width
        scaled_xmax = xmax * ratio_width
        scaled_ymin = ymin * ratio_height
        scaled_ymax = ymax * ratio_height
        scaled_box = torch.cat(
            (scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax), dim=-1
        )
        bbox = BoxList(scaled_box, size, mode="xyxy", ratios=self.ratios, img_id=self.img_id)
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.resize(size, *args, **kwargs)
            bbox.add_field(k, v)

        return bbox.convert(self.mode)

    def transpose(self, method):
        """
        Transpose bounding box (flip or rotate in 90 degree steps)
        :param method: One of :py:attr:`PIL.Image.FLIP_LEFT_RIGHT`,
          :py:attr:`PIL.Image.FLIP_TOP_BOTTOM`, :py:attr:`PIL.Image.ROTATE_90`,
          :py:attr:`PIL.Image.ROTATE_180`, :py:attr:`PIL.Image.ROTATE_270`,
          :py:attr:`PIL.Image.TRANSPOSE` or :py:attr:`PIL.Image.TRANSVERSE`.
        """
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        image_width, image_height = self.size
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if method == FLIP_LEFT_RIGHT:
            TO_REMOVE = 1
            transposed_xmin = image_width - xmax - TO_REMOVE
            transposed_xmax = image_width - xmin - TO_REMOVE
            transposed_ymin = ymin
            transposed_ymax = ymax
        elif method == FLIP_TOP_BOTTOM:
            transposed_xmin = xmin
            transposed_xmax = xmax
            transposed_ymin = image_height - ymax
            transposed_ymax = image_height - ymin

        transposed_boxes = torch.cat(
            (transposed_xmin, transposed_ymin, transposed_xmax, transposed_ymax), dim=-1
        )
        bbox = BoxList(transposed_boxes, self.size, mode="xyxy", ratios=self.ratios, img_id=self.img_id)
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.transpose(method)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def crop(self, box):
        """
        Cropss a rectangular region from this bounding box. The box is a
        4-tuple defining the left, upper, right, and lower pixel
        coordinate.
        """
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        w, h = box[2] - box[0], box[3] - box[1]
        cropped_xmin = (xmin - box[0]).clamp(min=0, max=w)
        cropped_ymin = (ymin - box[1]).clamp(min=0, max=h)
        cropped_xmax = (xmax - box[0]).clamp(min=0, max=w)
        cropped_ymax = (ymax - box[1]).clamp(min=0, max=h)

        # TODO should I filter empty boxes here?
        if False:
            is_empty = (cropped_xmin == cropped_xmax) | (cropped_ymin == cropped_ymax)

        cropped_box = torch.cat(
            (cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax), dim=-1
        )
        bbox = BoxList(cropped_box, (w, h), mode="xyxy", ratios=self.ratios, img_id=self.img_id)
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.crop(box)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    # Tensor-like methods

    def to(self, device):
        bbox = BoxList(self.bbox.to(device), self.size, self.mode, ratios=self.ratios, img_id=self.img_id)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            bbox.add_field(k, v)
        return bbox

    def __getitem__(self, item):
        bbox = BoxList(self.bbox[item], self.size, self.mode, ratios=self.ratios, img_id=self.img_id)
        for k, v in self.extra_fields.items():
            bbox.add_field(k, v[item])
        return bbox

    def __len__(self):
        return self.bbox.shape[0]

    def clip_to_image(self, remove_empty=True):
        TO_REMOVE = 1

        size = self.get_field('real_img_size') if self.has_field('real_img_size') else self.size

        self.bbox[:, 0].clamp_(min=0, max=size[0] - TO_REMOVE)
        self.bbox[:, 1].clamp_(min=0, max=size[1] - TO_REMOVE)
        self.bbox[:, 2].clamp_(min=0, max=size[0] - TO_REMOVE)
        self.bbox[:, 3].clamp_(min=0, max=size[1] - TO_REMOVE)
        if remove_empty:
            box = self.bbox
            keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
            return self[keep]
        return self

    def area(self):
        box = self.bbox
        if self.mode == "xyxy":
            TO_REMOVE = 1
            area = (box[:, 2] - box[:, 0] + TO_REMOVE) * (box[:, 3] - box[:, 1] + TO_REMOVE)
        elif self.mode == "xywh":
            area = box[:, 2] * box[:, 3]
        else:
            raise RuntimeError("Should not be here")

        return area

    def copy_with_fields(self, fields, skip_missing=False):
        bbox = BoxList(self.bbox, self.size, self.mode, ratios=self.ratios, img_id=self.img_id)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            if self.has_field(field):
                bbox.add_field(field, self.get_field(field))
            elif not skip_missing:
                raise KeyError("Field '{}' not found in {}".format(field, self))
        return bbox

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_boxes={}, ".format(len(self))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        s += "mode={})".format(self.mode)
        return s

def compute_corners(dimensions, alpha):
    dtype = dimensions.dtype

    num_boxes = dimensions.shape[0]
    h, w, l = torch.split( dimensions.view(num_boxes, 1, 3), [1, 1, 1], dim=2)
    unrot = torch.cat([torch.cat([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2], dim=2),
                       torch.cat([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2], dim=2)], dim=1)
    alpha_r = alpha.view(num_boxes, 1)

    x_rot_vect = torch.cat([torch.cos(alpha_r), torch.sin(alpha_r)], dim=1).view(num_boxes, 2, 1)
    x_rot = (unrot * x_rot_vect).sum(dim=1, keepdim=True)

    z_rot_vect = torch.cat([-torch.sin(alpha_r), torch.cos(alpha_r)], dim=1).view(num_boxes, 2, 1)
    z_rot = (unrot * z_rot_vect).sum(dim=1, keepdim=True)

    zeros = torch.zeros((num_boxes, 1, 1), dtype=dtype)
    if dimensions.is_cuda:
        zeros = zeros.cuda()
    y_rot = torch.cat([zeros, zeros, zeros, zeros, -h, -h, -h, -h], dim=2)

    corners_rot = torch.cat([x_rot, y_rot, z_rot], dim=1)
    return corners_rot

def compute_corners_sc(dimensions, sin, cos):
    dtype = dimensions.dtype

    num_boxes = dimensions.shape[0]
    h, w, l = torch.split( dimensions.view(num_boxes, 1, 3), [1, 1, 1], dim=2)
    unrot = torch.cat([torch.cat([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2], dim=2),
                       torch.cat([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2], dim=2)], dim=1)
    sin = sin.view(num_boxes, 1)
    cos = cos.view(num_boxes, 1)

    x_rot_vect = torch.cat([cos, sin], dim=1).view(num_boxes, 2, 1)
    x_rot = (unrot * x_rot_vect).sum(dim=1, keepdim=True, dtype=dtype)

    z_rot_vect = torch.cat([-sin, cos], dim=1).view(num_boxes, 2, 1)
    z_rot = (unrot * z_rot_vect).sum(dim=1, keepdim=True, dtype=dtype)

    zeros = torch.zeros((num_boxes, 1, 1), dtype=dtype)
    if dimensions.is_cuda:
        zeros = zeros.cuda()
    y_rot = torch.cat([zeros, zeros, zeros, zeros, -h, -h, -h, -h], dim=2)

    corners_rot = torch.cat([x_rot, y_rot, z_rot], dim=1)
    return corners_rot

def quan_to_angle(qw, qx, qy, qz):
    rx = torch.atan2(2.*(qw*qx + qy*qz), 1.-2.*(qx*qx + qy*qy))

    sinp = 2.*(qw*qy - qz*qx)
    sinp = sinp.clamp(-1., 1.)
    ry = torch.asin(sinp)

    rz = torch.atan2(2.*(qw*qz + qx*qy), 1.-2.*(qy*qy + qz*qz))

    return rx, ry, rz

def quan_to_rotation(q0, q1, q2, q3):
    x_rot = torch.stack([q0*q0 + q1*q1 - q2*q2 - q3*q3, 2*(q1*q2 - q0*q3), 2*(q0*q2 + q1*q3)], dim=1)
    y_rot = torch.stack([2*(q1*q2 + q0*q3), q0*q0 - q1*q1 + q2*q2 - q3*q3, 2*(q2*q3 - q0*q1)], dim=1)
    z_rot = torch.stack([2*(q1*q3 - q0*q2), 2*(q0*q1 + q2*q3), q0*q0 - q1*q1 - q2*q2 + q3*q3], dim=1)
    rot = torch.stack([x_rot, y_rot, z_rot], dim=1)
    return rot

def angle_to_quan(rx, ry, rz): # yaw (Z), pitch (Y), roll (X)
    cy = torch.cos(rz * 0.5);
    sy = torch.sin(rz * 0.5);
    cp = torch.cos(ry * 0.5);
    sp = torch.sin(ry * 0.5);
    cr = torch.cos(rx * 0.5);
    sr = torch.sin(rx * 0.5);

    qw = cy * cp * cr + sy * sp * sr
    qx = cy * cp * sr - sy * sp * cr
    qy = sy * cp * sr + cy * sp * cr
    qz = sy * cp * cr - cy * sp * sr

    return qw, qx, qy, qz

def project_rect_to_image(pts_3d_rect, P):
    n = pts_3d_rect.shape[0]
    ones = torch.ones((n,1))
    if pts_3d_rect.is_cuda:
        ones = ones.cuda()
    pts_3d_rect = torch.cat([pts_3d_rect, ones], dim=1)
    pts_2d = torch.mm(pts_3d_rect, torch.transpose(P, 0, 1)) # nx3
    pts_2d[:,0] /= pts_2d[:,2]
    pts_2d[:,1] /= pts_2d[:,2]
    return pts_2d[:,0:2]

class Box3DList(BoxList):
    def __init__(self, bbox, image_size, mode="xyxy", ratios=(1.,1.), box3d=None, Proj=None, img_id=None, Proj_R=None):
        super(Box3DList, self).__init__(bbox, image_size, mode, ratios, img_id=img_id)

        # 3D box in Camera Coordinate
        device = bbox.device if isinstance(bbox, torch.Tensor) else torch.device("cpu")
        if box3d is not None:
            box3d = torch.as_tensor(box3d, dtype=torch.float32, device=device)
            if box3d.ndimension() != 2:
                raise ValueError(
                    "box3d should have 2 dimensions, got {}".format(box3d.ndimension())
                )
            if box3d.size(-1) != 7:
                raise ValueError(
                    "last dimension of bbox should have a "
                    "size of 7, got {}".format(box3d.size(-1))
                )
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")
        self.box3d = box3d
        self.Proj = Proj
        self.Proj_R = Proj_R

    def box_corners(self):
        return compute_corners(self.box3d[:, :3], self.box3d[:, 6]).transpose(1,2)

    def box_center3ds(self):
        n = self.box3d.shape[0]
        zeros = torch.zeros((n, 1))
        if self.box3d.is_cuda:
            zeros = zeros.cuda()
        return self.box3d[:, 3:6] - \
            torch.cat([zeros, self.box3d[:, :1]/2., zeros], dim=1)

    def box_3dto2d(self):
        Proj = torch.as_tensor(self.Proj, dtype=torch.float32)
        if box_corners3d.is_cuda:
            Proj = Proj.cuda()
        return project_rect_to_image(self.box_center3ds, Proj)

    def box_corners2d(self, clip_image=True, return_right=False):
        box_corners3d = self.box_corners() + self.box3d[:, [3,4,5]][:, None, :]
        Proj = torch.as_tensor(self.Proj, dtype=torch.float32)
        if box_corners3d.is_cuda:
            Proj = Proj.cuda()
        box_corners2d = project_rect_to_image(box_corners3d.reshape(-1,3), Proj)
        box_corners2d = box_corners2d.reshape(-1, 8, 2)
        box_corners2d = torch.cat([box_corners2d.min(dim=1)[0], box_corners2d.max(dim=1)[0]], dim=1)
        if clip_image:
            box_corners2d[:, [0, 2]] = box_corners2d[:, [0, 2]].clamp(min=0., max=self.size[0])
            box_corners2d[:, [1, 3]] = box_corners2d[:, [1, 3]].clamp(min=0., max=self.size[1])

        if return_right:
            Proj_R = torch.as_tensor(self.Proj_R, dtype=torch.float32)
            if box_corners3d.is_cuda:
                Proj_R = Proj_R.cuda()
            box_corners2d_R = project_rect_to_image(box_corners3d.reshape(-1,3), Proj_R)
            box_corners2d_R = box_corners2d_R.reshape(-1, 8, 2)
            box_corners2d_R = torch.cat([box_corners2d_R.min(dim=1)[0], box_corners2d_R.max(dim=1)[0]], dim=1)
            if clip_image:
                box_corners2d_R[:, [0, 2]] = box_corners2d_R[:, [0, 2]].clamp(min=0., max=self.size[0])
                box_corners2d_R[:, [1, 3]] = box_corners2d_R[:, [1, 3]].clamp(min=0., max=self.size[1])
            return box_corners2d, box_corners2d_R
        return box_corners2d

    @classmethod
    def fromboxlist(cls, boxlist, box3d, Proj, Proj_R=None):
        assert isinstance(boxlist, BoxList), 'This method requires BoxList-type.'
        bbox = cls(boxlist.bbox, boxlist.size, mode=boxlist.mode, ratios=boxlist.ratios, box3d=box3d, Proj=Proj, img_id=boxlist.img_id, Proj_R=Proj_R)
        for k, v in boxlist.extra_fields.items():
            bbox.add_field(k, v)
        return bbox

    def crop(self, box):
        raise NotImplementedError()

    def transpose(self, method):
        assert method == FLIP_LEFT_RIGHT, 'Only support FLIP_LEFT_RIGHT'

        #flip bbox
        image_width, image_height = self.size
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        TO_REMOVE = 1
        transposed_xmin = image_width - xmax - TO_REMOVE
        transposed_xmax = image_width - xmin - TO_REMOVE
        transposed_ymin = ymin
        transposed_ymax = ymax

        transposed_boxes = torch.cat(
            (transposed_xmin, transposed_ymin, transposed_xmax, transposed_ymax), dim=-1
        )

        # flip box3ds
        focal_length = self.Proj[0, 0]
        ppoint_x = self.Proj[0, 2]
        trans_x = self.Proj[0, 3]
        h, w, l, x, y, z, alpha = torch.split(self.box3d, [1,1,1,1,1,1,1], 1)
        delta_x = (z * (image_width - 1 - 2 * ppoint_x) - 2 * trans_x) / focal_length - 2 * x
        x += delta_x
        alpha = ((alpha > 0).float() + (alpha <= 0).float() * -1) * np.pi - alpha
        box3d = torch.cat([h, w, l, x, y, z, alpha], 1)

        # print(alpha)

        bbox = Box3DList(transposed_boxes, self.size, mode="xyxy", ratios=self.ratios, box3d=box3d, Proj=self.Proj, img_id=self.img_id, Proj_R=self.Proj_R)
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.transpose(method)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def resize(self, size, *args, **kwargs):
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        self.ratios = ratios
        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            scaled_box = self.bbox * ratio
            bbox = Box3DList(scaled_box, size, mode=self.mode, ratios=self.ratios, box3d=self.box3d, Proj=self.Proj, img_id=self.img_id, Proj_R=self.Proj_R)
            # bbox._copy_extra_fields(self)
            for k, v in self.extra_fields.items():
                if not isinstance(v, torch.Tensor):
                    v = v.resize(size, *args, **kwargs)
                bbox.add_field(k, v)
            return bbox

        ratio_width, ratio_height = ratios
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        scaled_xmin = xmin * ratio_width
        scaled_xmax = xmax * ratio_width
        scaled_ymin = ymin * ratio_height
        scaled_ymax = ymax * ratio_height
        scaled_box = torch.cat(
            (scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax), dim=-1
        )
        bbox = Box3DList(scaled_box, size, mode="xyxy", ratios=self.ratios, box3d=self.box3d, Proj=self.Proj, img_id=self.img_id, Proj_R=self.Proj_R)
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.resize(size, *args, **kwargs)
            bbox.add_field(k, v)

        return bbox.convert(self.mode)

    def to(self, device):
        if self.box3d is not None:
            box3d = Box3DList(self.bbox.to(device), self.size, self.mode, ratios=self.ratios, box3d=self.box3d.to(device), Proj=self.Proj, img_id=self.img_id, Proj_R=self.Proj_R)
        else:
            box3d = Box3DList(self.bbox.to(device), self.size, self.mode, ratios=self.ratios, box3d=None, Proj=self.Proj, img_id=self.img_id, Proj_R=self.Proj_R)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            box3d.add_field(k, v)
        return box3d

    def __getitem__(self, item):
        if self.box3d is not None:
            bbox = Box3DList(self.bbox[item], self.size, self.mode, ratios=self.ratios, box3d=self.box3d[item], Proj=self.Proj, img_id=self.img_id, Proj_R=self.Proj_R)
        else:
            bbox = Box3DList(self.bbox[item], self.size, self.mode, ratios=self.ratios, box3d=None, Proj=self.Proj, img_id=self.img_id, Proj_R=self.Proj_R)
        for k, v in self.extra_fields.items():
            bbox.add_field(k, v[item])
        return bbox

    def convert(self, mode):
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")
        if mode == self.mode:
            return self
        # we only have two modes, so don't need to check
        # self.mode
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if mode == "xyxy":
            bbox = torch.cat((xmin, ymin, xmax, ymax), dim=-1)
            bbox = Box3DList(bbox, self.size, mode=mode, ratios=self.ratios, box3d=self.box3d, Proj=self.Proj, img_id=self.img_id, Proj_R=self.Proj_R)
        else:
            TO_REMOVE = 1
            bbox = torch.cat(
                (xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE), dim=-1
            )
            bbox = Box3DList(bbox, self.size, mode=mode, ratios=self.ratios, box3d=self.box3d, Proj=self.Proj, img_id=self.img_id, Proj_R=self.Proj_R)
        bbox._copy_extra_fields(self)
        return bbox

    def copy_with_fields(self, fields, skip_missing=False):
        bbox = Box3DList(self.bbox, self.size, self.mode, ratios=self.ratios, box3d=self.box3d, Proj=self.Proj, img_id=self.img_id, Proj_R=self.Proj_R)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            if self.has_field(field):
                bbox.add_field(field, self.get_field(field))
            elif not skip_missing:
                raise KeyError("Field '{}' not found in {}".format(field, self))
        return bbox

if __name__ == "__main__":
    bbox = BoxList([[0, 0, 10, 10], [0, 0, 5, 5]], (10, 10))
    s_bbox = bbox.resize((5, 5))
    print(s_bbox)
    print(s_bbox.bbox)

    t_bbox = bbox.transpose(0)
    print(t_bbox)
    print(t_bbox.bbox)

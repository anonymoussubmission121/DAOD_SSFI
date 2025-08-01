# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.utils import cat


class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(
        self, 
        proposal_matcher, 
        fg_bg_sampler, 
        box_coder, 
        cls_agnostic_bbox_reg=False
    ):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

    def match_targets_to_proposals(self, proposal, target, is_source=True):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Fast RCNN only need "labels" field for selecting the targets
        target = target.copy_with_fields("labels")
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        if not is_source:
            matched_targets = target[matched_idxs]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets, sample_for_da=False):
        labels = []
        regression_targets = []
        domain_labels = []
        regression_targets_unweighted = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            is_source = targets_per_image.get_field('is_source')
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image, is_source.any()
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, proposals_per_image.bbox
            )

            domain_label = torch.ones_like(labels_per_image, dtype=torch.uint8) if is_source.any() else torch.zeros_like(labels_per_image, dtype=torch.uint8)
            domain_labels.append(domain_label)

            if not is_source.any():
                labels_per_image[:] = 0
            if sample_for_da:
                labels_per_image[:] = 0

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)
            regression_targets_unweighted.append(matched_targets.bbox)

        return labels, regression_targets, domain_labels, regression_targets_unweighted

    def subsample(self, proposals, targets, num_gts):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """

        labels, regression_targets, domain_labels, matched_target_bbox = self.prepare_targets(proposals, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels, num_gts)

        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        for labels_per_image, regression_targets_per_image, proposals_per_image, domain_label_per_image, mtb_per_image in zip(
            labels, regression_targets, proposals, domain_labels, matched_target_bbox
        ):
            proposals_per_image.add_field("labels", labels_per_image)
            proposals_per_image.add_field(
                "regression_targets", regression_targets_per_image
            )
            proposals_per_image.add_field("domain_labels", domain_label_per_image)
            proposals_per_image.add_field("unweighted_targets", mtb_per_image)

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        pos_proposals_per_img = []
        neg_proposals_per_img = []
        filtered_pos_inds_imgs = []
        filtered_neg_inds_imgs = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            pos_inds = torch.nonzero(pos_inds_img).squeeze(1)
            neg_inds = torch.nonzero(neg_inds_img).squeeze(1)
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)   
                        
            filtered_pos_inds_img = []
            filtered_neg_inds_img = []
            for i in range(img_sampled_inds.shape[0]):
                if img_sampled_inds[i] in pos_inds:
                    filtered_pos_inds_img.append(i)
                elif img_sampled_inds[i] in neg_inds:
                    filtered_neg_inds_img.append(i)
                    
            filtered_pos_inds_imgs.append(filtered_pos_inds_img)
            filtered_neg_inds_imgs.append(filtered_neg_inds_img)

            pos_proposals_per_img.append(proposals[img_idx][pos_inds])
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image

        self._proposals = proposals
        return proposals, filtered_pos_inds_imgs, filtered_neg_inds_imgs
    
    def subsample_for_da(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """

        labels, _, domain_labels, _ = self.prepare_targets(proposals, targets, True)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        proposals = list(proposals)
        # add corresponding label and information to the bounding boxes
        for proposals_per_image, domain_label_per_image in zip(
            proposals, domain_labels
        ):
            proposals_per_image.add_field("domain_labels", domain_label_per_image)

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):              
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image

        return proposals

    def __call__(self, class_logits, box_regression, use_pseudo_labeling_weight='none'):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        class_logits = cat(class_logits, dim=0)
        box_regression = cat(box_regression, dim=0)
        device = class_logits.device

        if not hasattr(self, "_proposals"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposals

        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        regression_targets = cat(
            [proposal.get_field("regression_targets") for proposal in proposals], dim=0
        )

        domain_masks = cat([proposal.get_field("domain_labels") for proposal in proposals], dim=0)

        class_logits = class_logits[domain_masks, :]
        box_regression = box_regression[domain_masks, :]
        labels = labels[domain_masks]
        regression_targets = regression_targets[domain_masks, :]

        # import pdb
        # pdb.set_trace()
        # apply pseudo labeling weight strategy
        if use_pseudo_labeling_weight=='none':
            classification_loss = F.cross_entropy(class_logits, labels)
        elif use_pseudo_labeling_weight=='prob':
            ema_softmax = torch.softmax(class_logits.detach(), dim=1)
            pseudo_prob = torch.max(ema_softmax, dim=1)[0]
            ce = F.cross_entropy(class_logits, labels, reduction='none')
            classification_loss = torch.mean(pseudo_prob * ce)
        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]
        if self.cls_agnostic_bbox_reg:
            map_inds = torch.tensor([4, 5, 6, 7], device=device)
        else:
            map_inds = 4 * labels_pos[:, None] + torch.tensor(
                [0, 1, 2, 3], device=device)

        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds_subset[:, None], map_inds],
            regression_targets[sampled_pos_inds_subset],
            size_average=False,
            beta=1,
        )
        box_loss = box_loss / labels.numel()

        return classification_loss, box_loss, domain_masks


def make_roi_box_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    )

    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    loss_evaluator = FastRCNNLossComputation(
        matcher, 
        fg_bg_sampler, 
        box_coder, 
        cls_agnostic_bbox_reg
    )

    return loss_evaluator

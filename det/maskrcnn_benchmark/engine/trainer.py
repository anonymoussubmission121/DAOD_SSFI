# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist
import os
from maskrcnn_benchmark.utils.miscellaneous import mkdir

from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.engine.inference import inference

from maskrcnn_benchmark.structures.image_list import ImageList, to_image_list

from torchvision.transforms.functional import to_pil_image

from .visualization import plot_one
import maskrcnn_benchmark.data.transforms.transforms as Tr

def do_mask_da_train(
    model, model_teacher,
    source_data_loader,
    target_data_loader,
    masking,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    cfg,
    checkpointer_teacher,
    data_loader_val
):
    from maskrcnn_benchmark.structures.image_list import to_image_list


    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    logger.info("with_MIC: On")

    meters = MetricLogger(delimiter=" ")
    max_iter = len(source_data_loader)
    start_iter = arguments["iteration"]
    logger.info("start iteration: " + str(start_iter) + " running to max iterations: " + str(max_iter))

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)

    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)

    model.train()
    model_teacher.eval()

    start_training_time = time.time()
    end = time.time()

    transform_strong_src = Tr.Compose(
            [
                Tr.StrongAug(source=False),
                Tr.ToTensor(),
                Tr.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=cfg.INPUT.TO_BGR255),
            ]
        )
    
    transform_strong_tgt = Tr.Compose(
            [
                Tr.StrongAug(source=False),
                Tr.ToTensor(),
                Tr.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=cfg.INPUT.TO_BGR255),
            ]
        )
    
    transform_weak = Tr.Compose(
            [
                Tr.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=cfg.INPUT.TO_BGR255),
            ]
        )

    # track loss for plotting 
    training_losses = []
    masked_losses = []
    for iteration, ((_, source_images, source_targets, _), (_, target_images, target_targets, _)) in enumerate(zip(source_data_loader, target_data_loader), start_iter):
        model.train()
        model_teacher.eval()
        
        data_time = time.time() - end
        arguments["iteration"] = iteration

        source_images = source_images.to(device)
        target_images = target_images.to(device)

        source_targets = [target.to(device) for target in list(source_targets)]
        target_targets = [target.to(device) for target in list(target_targets)]

        # strong-weak augmentation of target images 
        target_images_weak = []
        target_targets_weak = []
        target_images_strong = []
        target_targets_strong = []
        for ori_img, ori_tgt in zip(target_images.tensors.clone().detach(), target_targets):
            img, tgt = transform_weak(ori_img, ori_tgt)
            target_images_weak.append(img)
            target_targets_weak.append(tgt)

            img, tgt = transform_strong_tgt(to_pil_image(ori_img), ori_tgt)
            target_images_strong.append(img)
            target_targets_strong.append(tgt)

        # strong-weak augmentation of source images
        source_images_weak = []
        source_targets_weak = []
        source_images_strong = []
        source_targets_strong = []
        for ori_img, ori_tgt in zip(source_images.tensors.clone().detach(), source_targets):
            img, tgt = transform_weak(ori_img, ori_tgt)
            source_images_weak.append(img)
            source_targets_weak.append(tgt)

            img, tgt = transform_strong_src(to_pil_image(ori_img), ori_tgt)
            source_images_strong.append(img)
            source_targets_strong.append(tgt)

        target_images_strong = to_image_list(torch.stack(target_images_strong).cuda())
        target_images_weak = to_image_list(torch.stack(target_images_weak).cuda())
        source_images_strong = to_image_list(torch.stack(source_images_strong).cuda())
        source_images_weak = to_image_list(torch.stack(source_images_weak).cuda())

        # mask the target image
        masked_target_images = masking(target_images_strong.tensors.clone().detach()).detach()

        # update teacher weights 
        model_teacher.update_weights(model, iteration)

        # get psuedolabels using weakly augmented target images
        target_output = model_teacher(target_images_weak)

        # process output to get pseudo masks 
        target_pseudo_labels, pseudo_masks = process_pred2label(target_output, threshold=cfg.MODEL.PSEUDO_LABEL_THRESHOLD)

        # supervised loss using strong and weak source images
        source_images = source_images_strong
        source_targets = source_targets_strong
        record_dict = model(source_images, source_targets, with_DA_ON=False, save_feats=True, contrast=False)
    
        weakaug_images = source_images_weak + target_images_weak
        weakaug_targets = source_targets_weak + target_targets_weak

        record_dict_da = model(weakaug_images, weakaug_targets, with_DA_ON=True, save_feats=True, contrast=False)

        new_record_da = {}
        for key in record_dict_da.keys():
            if 'da' in key:
                new_record_da[key] = record_dict_da[key]
            else:
                record_dict_da[key] = 0
        record_dict.update(new_record_da)
        
        # apply pseudo label on masked images
        if len(target_pseudo_labels)>0:
            masked_images = masked_target_images[pseudo_masks]
            masked_target = target_pseudo_labels

            sizes = []
            for img in masked_target:
                sizes.append((img.size[1], img.size[0]))
            
            # convert to image list with same size of masked target 
            masked_images = ImageList(masked_images, sizes)

            # student gets masked images 
            masked_loss_dict = model(masked_images, masked_target, 
                                     use_pseudo_labeling_weight=cfg.MODEL.PSEUDO_LABEL_WEIGHT, 
                                     with_DA_ON=False, save_feats=True, contrast=True)
            
            new_record_all_unlabel_data = {}
            for key in masked_loss_dict.keys():
                new_record_all_unlabel_data[key + "_mask"] = masked_loss_dict[
                    key
                ]
            record_dict.update(new_record_all_unlabel_data)

        # weight losses
        loss_dict = {}
        ml = 0
        for key in record_dict.keys():
            if key.startswith("loss"):
                if key == "loss_box_reg_mask" or key == "loss_rpn_box_reg_mask":
                    # pseudo bbox regression <- 0
                    loss_dict[key] = record_dict[key] * 0
                elif key.endswith('_mask') and 'da' in key:
                    loss_dict[key] = record_dict[key] * 0
                elif key == 'loss_classifier_mask' or key == 'loss_objectness_mask':
                    loss_dict[key] = record_dict[key] * cfg.MODEL.PSEUDO_LABEL_LAMBDA
                    ml += record_dict[key] * cfg.MODEL.PSEUDO_LABEL_LAMBDA
                else:  # supervised loss
                    loss_dict[key] = record_dict[key] * 1

        if ml != 0:
            ml = ml.cpu().detach().numpy()

        masked_losses.append(ml)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        # track losses 
        training_losses.append(losses_reduced.cpu().detach().numpy())

        optimizer.zero_grad()
        losses.backward()
        
        optimizer.step()
        scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0 and iteration >= 2000:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
            checkpointer_teacher.save("model_teacher_{:07d}".format(iteration), **arguments)

            dataset_name = cfg.DATASETS.TEST[0]
            output_folder_stu = os.path.join(cfg.OUTPUT_DIR, "inference_student_" + str(iteration), dataset_name)
            output_folder_tea = os.path.join(cfg.OUTPUT_DIR, "inference_teacher_" + str(iteration), dataset_name)
            mkdir(output_folder_stu)
            mkdir(output_folder_tea)

            inference(
                model,
                data_loader_val,
                dataset_name=dataset_name,
                iou_types=iou_types,
                box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=output_folder_stu,
            )

            inference(
                model_teacher,
                data_loader_val,
                dataset_name=dataset_name,
                iou_types=iou_types,
                box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=output_folder_tea,
            )

        if iteration == max_iter-1:
            checkpointer.save("model_final", **arguments)
            checkpointer_teacher.save("model_final_teacher", **arguments)
        if torch.isnan(losses_reduced).any():
            logger.critical('Loss is NaN, exiting...')
            return  

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

    plot_one(training_losses, "Per Forward Pass Training Loss", "Iterations", "Loss", "Training", cfg.OUTPUT_DIR)
    plot_one(masked_losses, "Per Forward Pass Masked Training Loss", "Iterations", "Loss", "Training", cfg.OUTPUT_DIR)


def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    cfg,
    data_loader_val,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)

    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)

    start_training_time = time.time()
    end = time.time()

    transform_strong = Tr.Compose(
            [
                Tr.StrongAug(source=False),
                Tr.ToTensor(),
                Tr.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=cfg.INPUT.TO_BGR255),
            ]
        )

    for iteration, (_, images, targets, idx) in enumerate(data_loader, start_iter):
        model.train()

        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        # perform strong augmentation
        strongaug_imgs = []
        strongaug_tgts = []
        for img, tgt, _ in zip(images.tensors.clone().detach(), targets, idx):
            img = to_pil_image(img)
            img, tgt = transform_strong(img, [tgt])
            strongaug_imgs.append(img)
            strongaug_tgts.append(tgt[0])
        strongaug_imgs = to_image_list(torch.stack(strongaug_imgs).cuda())
        images = strongaug_imgs
        targets = strongaug_tgts

        loss_dict = model(images, targets, with_DA_ON=False, save_feats=False, contrast=False)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

        if iteration % checkpoint_period == 0 and iteration >= 2500:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)

            dataset_name = cfg.DATASETS.TEST[0]
            output_folder_stu = os.path.join(cfg.OUTPUT_DIR, "inference_student_" + str(iteration), dataset_name)
            output_folder_tea = os.path.join(cfg.OUTPUT_DIR, "inference_teacher_" + str(iteration), dataset_name)
            mkdir(output_folder_stu)
            mkdir(output_folder_tea)

            inference(
                model,
                data_loader_val,
                dataset_name=dataset_name,
                iou_types=iou_types,
                box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=output_folder_stu,
            )

            if iteration % checkpoint_period == 0:
                checkpointer.save("model_{:07d}".format(iteration), **arguments)
            if iteration == max_iter:
                checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

def process_pred2label(target_output, threshold=0.7):
    from maskrcnn_benchmark.structures.bounding_box import BoxList

    pseudo_labels_list = []
    masks = []
    for idx, bbox_l in enumerate(target_output):
        pred_bboxes = bbox_l.bbox.detach()

        labels = bbox_l.get_field('labels').detach()
        scores = bbox_l.get_field('scores').detach()

        filtered_idx = scores>=threshold

        filtered_bboxes = pred_bboxes[filtered_idx]

        filtered_labels = labels[filtered_idx]

        new_bbox_list = BoxList(filtered_bboxes, bbox_l.size, mode=bbox_l.mode)

        new_bbox_list.add_field("labels", filtered_labels)

        domain_labels = torch.ones_like(filtered_labels, dtype=torch.uint8).to(filtered_labels.device)
        new_bbox_list.add_field("is_source", domain_labels)

        if len(new_bbox_list)>0:
            pseudo_labels_list.append(new_bbox_list)
            masks.append(idx)
    return pseudo_labels_list, masks

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os


class DatasetCatalog(object):
    DATA_DIR = "/opt/MIC/datasets"
    DATASETS = {
        "soda-a_train_coco": {
            "img_dir": "soda-a/train",
            "ann_file": "soda-a/annotations/train.json"
        },
        "soda-a_val_coco" : {
            "img_dir": "soda-a/val",
            "ann_file": "soda-a/annotations/val.json"
        },
        "soda-d_train_coco" : {
            "img_dir": "SODA-D/Img_Split/Images/train",
            "ann_file": "SODA-D/Img_Split/Annotations/train.json"
        },
        "soda-d_val_coco" : {
            "img_dir": "SODA-D/Img_Split/Images/val",
            "ann_file": "SODA-D/Img_Split/Annotations/val.json"
        },
        "coco_2017_train": {
            "img_dir": "coco/train2017",
            "ann_file": "coco/annotations/instances_train2017.json"
        },
        "coco_2017_val": {
            "img_dir": "coco/val2017",
            "ann_file": "coco/annotations/instances_val2017.json"
        },
        "coco_2014_train": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/instances_train2014.json"
        },
        "coco_2014_val": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_val2014.json"
        },
        "coco_2014_minival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_minival2014.json"
        },
        "coco_2014_valminusminival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_valminusminival2014.json"
        },
        "keypoints_coco_2014_train": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/person_keypoints_train2014.json",
        },
        "keypoints_coco_2014_val": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/person_keypoints_val2014.json"
        },
        "keypoints_coco_2014_minival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/person_keypoints_minival2014.json",
        },
        "keypoints_coco_2014_valminusminival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/person_keypoints_valminusminival2014.json",
        },
        "voc_2007_train": {
            "data_dir": "voc/VOC2007",
            "split": "train"
        },
        "voc_2007_train_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_train2007.json"
        },
        "voc_2007_val": {
            "data_dir": "voc/VOC2007",
            "split": "val"
        },
        "voc_2007_val_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_val2007.json"
        },
        "voc_2007_test": {
            "data_dir": "voc/VOC2007",
            "split": "test"
        },
        "voc_2007_test_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_test2007.json"
        },
        "voc_2012_train": {
            "data_dir": "voc/VOC2012",
            "split": "train"
        },
        "voc_2012_train_cocostyle": {
            "img_dir": "voc/VOC2012/JPEGImages",
            "ann_file": "voc/VOC2012/Annotations/pascal_train2012.json"
        },
        "voc_2012_val": {
            "data_dir": "voc/VOC2012",
            "split": "val"
        },
        "voc_2012_val_cocostyle": {
            "img_dir": "voc/VOC2012/JPEGImages",
            "ann_file": "voc/VOC2012/Annotations/pascal_val2012.json"
        },
        "voc_2012_test": {
            "data_dir": "voc/VOC2012",
            "split": "test"
            # PASCAL VOC2012 doesn't made the test annotations available, so there's no json annotation
        },
        "cityscapes_fine_instanceonly_seg_train_cocostyle": {
            "img_dir": "cityscapes/leftImg8bit",
            "ann_file": "cityscapes/annotations/instancesonly_filtered_gtFine_train_poly.json"
        },
        "cityscapes_fine_instanceonly_seg_val_cocostyle": {
            "img_dir": "cityscapes/leftImg8bit",
            "ann_file": "cityscapes/annotations/instancesonly_filtered_gtFine_val_poly.json"
        },
        "cityscapes_fine_instanceonly_seg_test_cocostyle": {
            "img_dir": "cityscapes/leftImg8bit",
            "ann_file": "cityscapes/annotations/instancesonly_filtered_gtFine_test_poly.json"
        },
        "cityscapes_train_coco" : {
            "img_dir": "cityscapes/leftImg8bit/train",
            "ann_file": "cityscapes/annotations/cityscapes_train.json"
        },
        "cityscapes_train_coco_aldi" : {
            "img_dir": "cityscapes/leftImg8bit/train",
            "ann_file": "cityscapes/annotations/cityscapes_train_instances.json"
        },
        "cityscapes_ip2p_train_coco" : {
            "img_dir": "cityscapes/leftImg8bit_foggy_ip2p/train",
            "ann_file": "cityscapes/annotations/cityscapes_train_ip2p.json"
        },
        "cityscapes_val_coco" : {
            "img_dir": "cityscapes/leftImg8bit/val",
            "ann_file": "cityscapes/annotations/cityscapes_val.json"
        },
        "cityscapes_val_coco_aldi" : {
            "img_dir": "cityscapes/leftImg8bit/val",
            "ann_file": "cityscapes/annotations/cityscapes_val_instances.json"
        },
        "cityscapes_train_coco_small" : {
            "img_dir": "cityscapes/leftImg8bit/train",
            "ann_file": "cityscapes/annotations/cityscapes_train_small.json"
        },
        "cityscapes_val_coco_small" : {
            "img_dir": "cityscapes/leftImg8bit/val",
            "ann_file": "cityscapes/annotations/cityscapes_val_small.json"
        },
        "foggy_cityscapes_fine_instanceonly_seg_train_cocostyle": {
            "img_dir": "foggy_cityscapes/leftImg8bit_foggy",
            "ann_file": "foggy_cityscapes/annotations/foggy_instancesonly_filtered_gtFine_train_poly.json"
        },
        "foggy_cityscapes_fine_instanceonly_seg_val_cocostyle": {
            "img_dir": "foggy_cityscapes/leftImg8bit_foggy",
            "ann_file": "foggy_cityscapes/annotations/foggy_instancesonly_filtered_gtFine_val_poly.json"
        },
        "bdd_snowy_train_coco": {
            "img_dir": "bdd_daytime/images/100k/train",
            "ann_file": "bdd_daytime/annotations/bdd_daytime_train_snowy.json"
        },
        "bdd_snowy_val_coco": {
            "img_dir": "bdd_daytime/images/100k/val",
            "ann_file": "bdd_daytime/annotations/bdd_daytime_val_snowy.json"
        },
        "bdd_rainy_train_coco": {
            "img_dir": "bdd_daytime/images/100k/train",
            "ann_file": "bdd_daytime/annotations/bdd_daytime_train_rainy.json"
        },
        "bdd_rainy_val_coco": {
            "img_dir": "bdd_daytime/images/100k/val",
            "ann_file": "bdd_daytime/annotations/bdd_daytime_val_rainy.json"
        },
        "bdd_daytime_train_coco": {
            "img_dir": "bdd_daytime/images/100k/train",
            "ann_file": "bdd_daytime/annotations/bdd_daytime_train.json"
        },
        "bdd_daytime_val_coco": {
            "img_dir": "bdd_daytime/images/100k/val",
            "ann_file": "bdd_daytime/annotations/bdd_daytime_val.json"
        },
        "bdd_daytime_train_coco_small": {
            "img_dir": "bdd_daytime/images/100k/train",
            "ann_file": "bdd_daytime/annotations/bdd_daytime_train_small.json"
        },
        "bdd_daytime_val_coco_small": {
            "img_dir": "bdd_daytime/images/100k/val",
            "ann_file": "bdd_daytime/annotations/bdd_daytime_val_small.json"
        },
        "foggy_cityscapes_train_coco": {
            "img_dir": "cityscapes/leftImg8bit_foggy/train",
            "ann_file": "cityscapes/annotations/foggy_cityscapes_train.json"
        }, 
        "foggy_cityscapes_train_coco_small": {
            "img_dir": "cityscapes/leftImg8bit_foggy/train",
            "ann_file": "cityscapes/annotations/foggy_cityscapes_train_small.json"
        }, 
        "foggy_cityscapes_train_coco_small_defog": {
            "img_dir": "cityscapes/leftImg8bit_defog/train",
            "ann_file": "cityscapes/annotations/foggy_cityscapes_train_small_defog.json"
        }, 
        "foggy_cityscapes_val_coco": {
            "img_dir": "cityscapes/leftImg8bit_foggy/val",
            "ann_file": "cityscapes/annotations/foggy_cityscapes_val.json"
        }, 
        "foggy_cityscapes_val_coco_aldi": {
            "img_dir": "cityscapes/leftImg8bit_foggy/val",
            "ann_file": "cityscapes/annotations/cityscapes_val_instances_foggyALL_ids.json"
        }, 
        "foggy_cityscapes_val_coco_small": {
            "img_dir": "cityscapes/leftImg8bit_foggy/val",
            "ann_file": "cityscapes/annotations/foggy_cityscapes_val_small.json"
        },
        "foggy_cityscapes_val_coco_small_defog": {
            "img_dir": "cityscapes/leftImg8bit_defog/val",
            "ann_file": "cityscapes/annotations/foggy_cityscapes_val_small_defog.json"
        },
        "shuttle_annarbor_train_coco" : {
            "img_dir": "Shuttle_Deepen_Dataset/AnnArbor/train",
            "ann_file": "Shuttle_Deepen_Dataset/AnnArbor/annotations/train.json"
        },
        "shuttle_annarbor_val_coco" : {
            "img_dir": "Shuttle_Deepen_Dataset/AnnArbor/val",
            "ann_file": "Shuttle_Deepen_Dataset/AnnArbor/annotations/val.json"
        },
        "shuttle_mcs_train_coco" : {
            "img_dir": "Shuttle_Deepen_Dataset/Detroit/train",
            "ann_file": "Shuttle_Deepen_Dataset/Detroit/annotations/train.json"
        },
        "shuttle_mcs_val_coco" : {
            "img_dir": "Shuttle_Deepen_Dataset/Detroit/val",
            "ann_file": "Shuttle_Deepen_Dataset/Detroit/annotations/val.json"
        },
        "shuttle_mcity_train_coco" : {
            "img_dir": "Shuttle_Deepen_Dataset/MCity/val",
            "ann_file": "Shuttle_Deepen_Dataset/MCity/annotations/val.json"
        },
        "shuttle_mcity_val_coco" : {
            "img_dir": "Shuttle_Deepen_Dataset/AnnArbor/val",
            "ann_file": "Shuttle_Deepen_Dataset/AnnArbor/annotations/val.json"
        },
        "dawn_train_coco": {
            "img_dir": "DAWN/Train",
            "ann_file": "DAWN/annotations/dawn_train.json"
        },
        "dawn_val_coco": {
            "img_dir": "DAWN/Val",
            "ann_file": "DAWN/annotations/dawn_val.json"
        },
        "dawn_2_train_coco": {
            "img_dir": "DAWN_2/Train",
            "ann_file": "DAWN_2/annotations/dawn_train.json"
        },
        "dawn_2_val_coco": {
            "img_dir": "DAWN_2/Val",
            "ann_file": "DAWN_2/annotations/dawn_val.json"
        },
        'sim10k_cocostyle': {
                "img_dir": 'sim10K/JPEGImages',
                "ann_file": 'sim10K/car_instances.json'
        },
        'kitti_cocostyle': {
                "img_dir": 'kitti/training/image_2',
                "ann_file": 'kitti/annotations/caronly_training.json'
        },
        'cityscapes_car_train_cocostyle': {
                "img_dir": 'cityscapes/leftImg8bit/train',
                "ann_file": 'cityscapes/annotations/caronly_filtered_gtFine_train.json',
        },
        'cityscapes_car_val_cocostyle': {
                "img_dir": 'cityscapes/leftImg8bit/val',
                "ann_file": 'cityscapes/annotations/caronly_filtered_gtFine_val.json',
        },
        "voc_2007_train_watercolor_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/annotations/pascal_train2007.json"
        },
        "voc_2007_val_watercolor_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/annotations/pascal_val2007.json"
        },
        "voc_2012_train_watercolor_cocostyle": {
            "img_dir": "voc/VOC2012/JPEGImages",
            "ann_file": "voc/annotations/pascal_train2012.json"
        },
        "voc_2012_val_watercolor_cocostyle": {
            "img_dir": "voc/VOC2012/JPEGImages",
            "ann_file": "voc/annotations/Annotations/pascal_val2012.json"
        },
        'clipart_cocostyle': {
                "img_dir": 'clipart/JPEGImages',
                "ann_file": 'clipart/instances.json',
        },
        'watercolor_train_cocostyle': {
                "img_dir": 'watercolor/JPEGImages',
                "ann_file": 'watercolor/instances_train.json',
        },
        'watercolor_val_cocostyle': {
                "img_dir": 'watercolor/JPEGImages',
                "ann_file": 'watercolor/instances_test.json',
        }
    }

    @staticmethod
    def get(name):
        print(name)
        if "coco" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="COCODataset",
                args=args,
            )
        elif "voc" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(
                factory="PascalVOCDataset",
                args=args,
            )
        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog(object):
    S3_C2_DETECTRON_URL = "https://dl.fbaipublicfiles.com/detectron"
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-50-GN": "ImageNetPretrained/47261647/R-50-GN.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "MSRA/R-101-GN": "ImageNetPretrained/47592356/R-101-GN.pkl",
        "FAIR/20171220/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
    }

    C2_DETECTRON_SUFFIX = "output/train/{}coco_2014_train%3A{}coco_2014_valminusminival/generalized_rcnn/model_final.pkl"
    C2_DETECTRON_MODELS = {
        "35857197/e2e_faster_rcnn_R-50-C4_1x": "01_33_49.iAX0mXvW",
        "35857345/e2e_faster_rcnn_R-50-FPN_1x": "01_36_30.cUF7QR7I",
        "35857890/e2e_faster_rcnn_R-101-FPN_1x": "01_38_50.sNxI7sX7",
        "36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x": "06_31_39.5MIHi1fZ",
        "35858791/e2e_mask_rcnn_R-50-C4_1x": "01_45_57.ZgkA7hPB",
        "35858933/e2e_mask_rcnn_R-50-FPN_1x": "01_48_14.DzEQe4wC",
        "35861795/e2e_mask_rcnn_R-101-FPN_1x": "02_31_37.KqyEK4tT",
        "36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x": "06_35_59.RZotkLKI",
        "37129812/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x": "09_35_36.8pzTQKYK",
        # keypoints
        "37697547/e2e_keypoint_rcnn_R-50-FPN_1x": "08_42_54.kdzV35ao"
    }

    @staticmethod
    def get(name):
        if name.startswith("Caffe2Detectron/COCO"):
            return ModelCatalog.get_c2_detectron_12_2017_baselines(name)
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog {}".format(name))

    @staticmethod
    def get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        name = name[len("ImageNetPretrained/"):]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        url = "/".join([prefix, name])
        return url

    @staticmethod
    def get_c2_detectron_12_2017_baselines(name):
        # Detectron C2 models are stored following the structure
        # prefix/<model_id>/2012_2017_baselines/<model_name>.yaml.<signature>/suffix
        # we use as identifiers in the catalog Caffe2Detectron/COCO/<model_id>/<model_name>
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        dataset_tag = "keypoints_" if "keypoint" in name else ""
        suffix = ModelCatalog.C2_DETECTRON_SUFFIX.format(dataset_tag, dataset_tag)
        # remove identification prefix
        name = name[len("Caffe2Detectron/COCO/"):]
        # split in <model_id> and <model_name>
        model_id, model_name = name.split("/")
        # parsing to make it match the url address from the Caffe2 models
        model_name = "{}.yaml".format(model_name)
        signature = ModelCatalog.C2_DETECTRON_MODELS[name]
        unique_name = ".".join([model_name, signature])
        url = "/".join([prefix, model_id, "12_2017_baselines", unique_name, suffix])
        return url

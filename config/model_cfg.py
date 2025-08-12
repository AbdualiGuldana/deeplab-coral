from type import (
    MODELTYPE,
    DEEPLABTYPE,
    MASKRCNNTYPE,
)

from entity import (
    Deeplabv3Entity,
    MaskRcnnEntity,
    DataEntity,
    DataloaderEntity,
    TrainEntity,
)


classes = 6 # 배경포함 6개
DEVICE = "cuda:0"
MODEL_TYPE = MODELTYPE.DEEPLABV3
SENSOR_RATIO = 0.3
RESIZE_SCALE = 65 # backbone feature map width, height

DATA_CFG = DataEntity(
    root_path="/home/allbigdat/data",
    image_base_path="images/",
    train_anno_path="COCO/train_without_street.json",
    valid_anno_path="COCO/valid_without_street.json",
    test_anno_path="COCO/test_without_street.json",
)



TRAIN_DATALOADER_CFG = DataloaderEntity(
    bathc_size=16,
    num_worker=0,
    shuffle=True
)

VALID_DATALOADER_CFG = DataloaderEntity(
    bathc_size=16,
    num_worker=0,
    shuffle=False
)

TEST_DATALOADER_CFG = DataloaderEntity(
    bathc_size=16,
    num_worker=0,
    shuffle=False
)


TRAIN_CFG = TrainEntity(
    accum_step=1,
    num_epoch=50,
    log_step=20,
)

TRAIN_PIPE = [
    dict(type="ToPILImage", params=dict()),
    dict(type="Resize", params=dict(size=(520, 520))),
    # Start of new augmentations
    dict(type="RandomHorizontalFlip", params=dict(p=0.5)),
    dict(type="RandomVerticalFlip", params=dict(p=0.5)),
    dict(type="RandomRotation", params=dict(degrees=10)),
    dict(type="ColorJitter", params=dict(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)),
    # End of new augmentations
    dict(type="ToTensor", params=dict()),
    # It is highly recommended to include a Normalize transformation
    # as most pretrained models are trained on normalized data.
    # You should uncomment this and use the mean/std for ImageNet
    # or your own dataset.
    dict(type="Normalize", params=dict(std=[0.229, 0.224, 0.225], mean=[0.485, 0.456, 0.406])),
]




'''

# TODO: Augmentation 고려
TRAIN_PIPE = [
    dict(type="ToPILImage", params=dict()),
    dict(type="Resize", params=dict(size=(520, 520))),
    dict(type="ToTensor", params=dict()),
    # dict(type="Normalize", params=dict(std=[0.229, 0.224, 0.225], mean=[0.485, 0.456, 0.406])),
]
# TRAIN_PIPE = [
#     dict(type="ToTensor", params=dict()),
#     dict(type="Resize", params=dict(size=(520, 520))),
#     dict(type="Normalize", params=dict(std=[0.229, 0.224, 0.225], mean=[0.485, 0.456, 0.406])),
# ]

'''

# TEST_PIPE = [
#     dict(type="ToTensor", params=dict()),
#     dict(type="Resize", params=dict(size=(520, 520))),
#     dict(type="Normalize", params=dict(std=[0.229, 0.224, 0.225], mean=[0.485, 0.456, 0.406])),
# ]

# In VALID_PIPE / TEST_PIPE add Normalize with ImageNet stats
VALID_PIPE = [
    dict(type="ToPILImage", params=dict()),
    dict(type="Resize", params=dict(size=(520, 520))),
    dict(type="ToTensor", params=dict()),
    dict(type="Normalize", params=dict(std=[0.229, 0.224, 0.225], mean=[0.485, 0.456, 0.406])),
]

TEST_PIPE = [
    dict(type="ToPILImage", params=dict()),
    dict(type="Resize", params=dict(size=(520, 520))),
    dict(type="ToTensor", params=dict()),
    dict(type="Normalize", params=dict(std=[0.229, 0.224, 0.225], mean=[0.485, 0.456, 0.406])),
]



# VALID_PIPE = [
#     dict(type="ToPILImage", params=dict()),
#     dict(type="Resize", params=dict(size=(520, 520))),
#     dict(type="ToTensor", params=dict()),
# ]


# TEST_PIPE = [
#     dict(type="ToPILImage", params=dict()),
#     dict(type="Resize", params=dict(size=(520, 520))),
#     dict(type="ToTensor", params=dict()),
#     # dict(type="Normalize", params=dict(std=[0.229, 0.224, 0.225], mean=[0.485, 0.456, 0.406])),
# ]

MODEL_CFG = {
    MODELTYPE.DEEPLABV3: Deeplabv3Entity(
        type=DEEPLABTYPE.RESNET50,
        num_classes=classes,
        use_cbam=True,
        params=dict(
            pretrained=True,
        ),
        #load_from="/home/dan/Desktop/deeplab-sensor-fusion/output _original/1.pth" #actually best.pth renamed 50.pth
        #load_from = None
        
        #load_from = "/home/dan/Desktop/deeplab-sensor-fusion/output/25.pth"
        #load_from = '/home/dan/Desktop/deeplab-sensor-fusion/output_output/25.pth'
        load_from = '/home/dan/Desktop/deeplab-sensor-fusion/output_0out/14.pth'
    ),
    MODELTYPE.MASKRCNN: MaskRcnnEntity(
        type=MASKRCNNTYPE.RESNET50V2,
        params=dict(
            num_classes=classes
        ),
    )
}[MODEL_TYPE]

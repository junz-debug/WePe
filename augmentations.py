from torchvision import transforms

from .my_transforms import make_normalize_transform


class DataAugmentationDINO:
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        self.source_trans = transforms.Compose(
            [
                transforms.Resize((global_crops_size, global_crops_size)),
                transforms.CenterCrop(global_crops_size),
                transforms.ToTensor(),
                make_normalize_transform(),
            ]
        )

    def __call__(self, image):
        return {"source1": self.source_trans(image)}


from detectron2.data import DatasetMapper
from detectron2.data import detection_utils as utils
from detectron2.data import build_detection_test_loader
import torch


class CustomDatasetMapper(DatasetMapper):
    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        image_shape = dataset_dict['height'], dataset_dict['width']
        annos =  dataset_dict["annotations"]
        mislabeled = [obj["mislabeled"] for obj in annos]

        instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )
        instances.gt_mislabeled = torch.tensor(mislabeled, dtype = torch.bool)  
        dataset_dict = super(CustomDatasetMapper, self).__call__(dataset_dict)
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
    
        return dataset_dict


def build_test_loader(cfg, dataset_name):
    dataset_mapper = CustomDatasetMapper(cfg, False)
    return build_detection_test_loader(cfg, dataset_name, dataset_mapper)


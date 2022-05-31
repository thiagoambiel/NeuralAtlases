import logging
import numpy as np

try:
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
except ImportError:
    logging.basicConfig(
        format='%(asctime)s :: %(levelname)s :: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.getLogger().setLevel(logging.INFO)
    logging.warning("Can't import 'detectron2' library, MaskRCNN disabled!")


class MaskRCNN:
    def __init__(self, class_name: str):
        self.model_url = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

        try:
            self.cfg = get_cfg()
        except NameError:
            logging.error("The 'MaskRCNN' algorithm is disabled. "
                          "Use 'MODNet' or install the 'detectron2' library.")
            exit()

        self.cfg.merge_from_file(model_zoo.get_config_file(self.model_url))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.model_url)

        self.model = DefaultPredictor(self.cfg)

        self.class_name = class_name

    def __call__(self, img: np.array):
        outputs = self.model(img)

        mask = None

        if self.class_name == 'anything':
            try:
                mask = outputs["instances"].pred_masks[0].cpu().numpy() * 255
            except:
                mask = np.zeros((img.shape[0], img.shape[1]))

        else:
            for instance_idx in range(len(outputs["instances"])):
                instance = outputs["instances"][instance_idx]

                class_idx = (instance.pred_classes.cpu()).long()
                class_label = self.model.metadata.thing_classes[class_idx]

                if class_label == self.class_name:
                    mask = outputs["instances"].pred_masks[instance_idx].cpu().numpy() * 255
                    break

            if mask is None:
                mask = np.zeros((img.shape[0], img.shape[1]))

        return mask

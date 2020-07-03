from pathlib import Path
from typing import Union

import cv2
import dlib
from keras.utils.data_utils import get_file
import numpy as np

from wide_resnet import WideResNet

PRETRAINED_MODEL = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.28-3.73.hdf5"
MODHASH = 'fbe63257a054c1c5466cfd7bf14646d6'


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), 1)
    h, w, _ = img.shape
    r = 640 / max(w, h)
    return cv2.resize(img, (int(w * r), int(h * r)))


class GenderAnnotator:
    def __init__(self):
        # Constants
        self.img_size = 64
        self.depth = 16
        self.k = 8
        self.margin = 0.4

        # Model and Weights
        self.weight_file = get_file("weights.28-3.73.hdf5",
                                    PRETRAINED_MODEL,
                                    cache_subdir="pretrained_models",
                                    file_hash=MODHASH,
                                    cache_dir=str(
                                        Path(__file__).resolve().parent))
        self.model = WideResNet(self.img_size, depth=self.depth, k=self.k)()
        self.model.load_weights(self.weight_file)

        # Face Detector
        self.detector = dlib.get_frontal_face_detector()

    def estimate_gender(self, path: Path) -> Union[float, None]:
        img = load_image(path)
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(input_img)

        # detect faces using dlib detector
        detected = self.detector(input_img, 1)
        face = np.empty((1, self.img_size, self.img_size, 3))

        if len(detected) > 0:
            d = detected[0]
            x1, y1, x2, y2, w, h = d.left(), d.top(
            ), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            xw1 = max(int(x1 - self.margin * w), 0)
            yw1 = max(int(y1 - self.margin * h), 0)
            xw2 = min(int(x2 + self.margin * w), img_w - 1)
            yw2 = min(int(y2 + self.margin * h), img_h - 1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
            face[0, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :],
                                          (self.img_size, self.img_size))

            # predict gender of the detected face
            results = self.model.predict(face)
            predicted_genders = results[0]
            return predicted_genders[0][1]  # manliness
        else:
            return None


if __name__ == "__main__":
    test_path = Path(__file__).resolve().parent / 'test'
    test_0 = test_path / 'test_0.jpeg'
    test_1 = test_path / 'test_1.jpeg'

    annotator = GenderAnnotator()
    test = annotator.estimate_gender(test_0)
    test = annotator.estimate_gender(test_1)
    print(f'Manliness {round(test*100,1)}%')

import os
import cv2
import numpy as np

_IMAGE_SIZE = 224
_IMAGE_MIN_SIZE = 256
_SAMPLE_VIDEO_FRAMES = 79

foldersPath = '../datasets/ava/frames'
folders = os.listdir(foldersPath)


def crop_center(img, crop_width, crop_height):
    h, w, _ = img.shape
    startw = w//2 - (crop_width//2)
    starth = h//2 - (crop_height//2)
    return img[starth:starth+crop_height, startw:startw+crop_width, :]


def run():
    if not os.path.exists('data/ava'):
        os.mkdir('data/ava')

    for folder in folders:
        print(folder)
        output = []
        folder_path = os.path.join(foldersPath, folder)
        image_names = os.listdir(folder_path)
        _SAMPLE_VIDEO_FRAMES = len(image_names)

        for i in range(_SAMPLE_VIDEO_FRAMES):
            imageName = '%03d.jpg' % (i + 1)
            img = cv2.imread(os.path.join(folder_path, imageName))
            height, width, _ = img.shape
            if width < height:
                good_width = _IMAGE_MIN_SIZE
                good_height = int(height/(float(width)/good_width))
            else:
                good_height = _IMAGE_MIN_SIZE
                good_width = int(width/(float(height)/good_height))

            img = cv2.resize(img, dsize=(good_width, good_height), interpolation=cv2.INTER_LINEAR)
            img = img / float(img.max()) * 2 - 1.0
            img = crop_center(img, _IMAGE_SIZE, _IMAGE_SIZE)

            output.append(img)

        data_array = np.array(output)
        data_array = np.expand_dims(data_array, axis=0)
        np.save('data/ava/'+folder, data_array)


if __name__ == '__main__':
    run()

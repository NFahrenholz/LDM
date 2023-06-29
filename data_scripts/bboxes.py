import numpy as np
import glob as glob
import cv2
import xml.etree.ElementTree as ET

from scipy import ndimage
from tqdm import trange
from skimage.measure import label, regionprops, find_contours

from config import config

def write_xml(bboxes, labels, name):
    data = ET.Element('data')

    for bbox, label in zip(bboxes, labels):
        bndbox = ET.SubElement(data, 'bbox')

        label_id = ET.SubElement(bndbox, 'label_id')
        label_name = ET.SubElement(bndbox, 'label_name')

        x1 = ET.SubElement(bndbox, 'x1')
        y1 = ET.SubElement(bndbox, 'y1')
        x2 = ET.SubElement(bndbox, 'x2')
        y2 = ET.SubElement(bndbox, 'y2')

        label_id.text = str(label)
        label_name.text = config.classes[label]
        x1.text = str(bbox[0])
        y1.text = str(bbox[1])
        x2.text = str(bbox[2])
        y2.text = str(bbox[3])

    tree = ET.ElementTree(data)
    tree.write(f"{config.output_path}/{name}.xml")


def get_bboxes(mask):
    obj_ids = np.unique(mask)
    bboxes = []
    labels = []

    for obj in obj_ids:
        bin_mask = (mask == obj).astype(int)
        structure = np.ones((3, 3))
        padded_bin_mask = ndimage.binary_dilation(bin_mask, structure=structure, iterations=30).astype(int)
        label, num = ndimage.label(padded_bin_mask, structure=structure)
        label = label * bin_mask
        props = regionprops(label)
        for prop in props:
            x1 = prop.bbox[1]
            y1 = prop.bbox[0]

            x2 = prop.bbox[3]
            y2 = prop.bbox[2]

            bboxes.append((x1, y1, x2, y2))
            labels.append(obj)

    return bboxes, labels



if __name__ == '__main__':
    image_paths = []
    image_names = []

    mask_paths = []

    for video in config.videos:
        i_ps = sorted(glob.glob(f"{config.data_path}/{video}/Images/*.png"))
        i = sorted([i_p.split('/')[-1] for i_p in i_ps])

        image_paths += i_ps
        image_names += i

        m_ps = sorted(glob.glob(f"{config.data_path}/{video}/Labels/*.png"))

        mask_paths += m_ps

    for id in trange(len(image_paths)):
        mask = cv2.imread(mask_paths[id], cv2.IMREAD_GRAYSCALE)

        bboxes, labels = get_bboxes(mask)

        write_xml(bboxes, labels, image_names[id])
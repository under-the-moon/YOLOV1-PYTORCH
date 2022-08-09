import os
import xml.etree.ElementTree as ET


class DataProcess:

    def __init__(self, data_path, names, save_path):
        """
        :param data_path:
                     parent_dir:
                        --VOC2007
                        --VOC2012

        :param names:
        """
        self.data_path = data_path
        self.names = names
        self.save_path = save_path

    def process(self):
        out_file = open(self.save_path, 'w')
        dirnames = os.listdir(self.data_path)
        for dirname in dirnames:
            path = os.path.join(self.data_path, dirname)
            anno_path = os.path.join(path, 'Annotations')
            img_path = os.path.join(path, 'JPEGImages')

            train_txt = os.path.join(path, 'ImageSets', 'Main', 'train.txt')
            val_txt = os.path.join(path, 'ImageSets', 'Main', 'val.txt')
            if not os.path.exists(train_txt):
                raise ValueError(f'{train_txt} not existed ! please check it !')

            lines = open(train_txt).readlines()
            self._process(lines, anno_path, img_path, out_file)

            if os.path.exists(val_txt):
                lines = open(val_txt).readlines()
                self._process(lines, anno_path, img_path, out_file)

    def _process(self, lines, anno_path, img_path, out_file):
        for line in lines:
            line = line.strip()
            xml_file = os.path.join(anno_path, line + '.xml')
            img_file = os.path.join(img_path, line + '.jpg')
            if not os.path.exists(xml_file) or not os.path.exists(img_file):
                continue

            out_file.write(img_file)

            tree = ET.parse(xml_file)
            root = tree.getroot()

            for obj in root.iter('object'):
                if int(obj.find('difficult').text) == 1:
                    continue
                cls = obj.find('name').text
                xmlbox = obj.find('bndbox')
                bb = [int(xmlbox.find(x).text) for x in ('xmin', 'ymin', 'xmax', 'ymax')]
                cls_id = self.names.index(cls.lower())  # class id
                out_file.write(" " + ",".join([str(a) for a in [*bb, cls_id]]))
            out_file.write('\n')

import os
import copy
import cv2
from tqdm import tqdm
from xml.dom import minidom as dom
import xml.etree.ElementTree as ET


class XMLGenerator(object):
    def __init__(self, xml_name: str):
        self.doc = dom.Document()
        self.xml_name = xml_name

    def create_append_node(self, node_name, root_node=None):
        """创建一个新node并将node添加到root_node下"""
        new_node = self.doc.createElement(node_name)
        if root_node is not None:
            root_node.appendChild(new_node)
        else:
            self.doc.appendChild(new_node)
        return new_node

    def create_text_node(self, node_name, node_value, root_node):
        """
        创建一个新node，然后在该node中添加一个text_node，
        最后将node添加到root_node下
        """
        new_node = self.doc.createElement(node_name)
        node_data = self.doc.createTextNode(node_value)
        new_node.appendChild(node_data)
        root_node.appendChild(new_node)

    def create_object_node(self, info_dict: dict = None, root_node: str = None):
        if (info_dict is None) or (root_node is None):
            return

        object_node = self.create_append_node('object', root_node)
        box_node = self.create_append_node('bndbox', object_node)
        self.create_text_node("xmin", info_dict.pop("xmin"), box_node)
        self.create_text_node("ymin", info_dict.pop("ymin"), box_node)
        self.create_text_node("xmax", info_dict.pop("xmax"), box_node)
        self.create_text_node("ymax", info_dict.pop("ymax"), box_node)

        for k, v in info_dict.items():
            self.create_text_node(k, v, object_node)

    def save_xml(self):
        f = open(self.xml_name, "w")
        self.doc.writexml(f, addindent="\t", newl="\n")
        f.close()


def create_pascal_voc_xml(filename: str = None,
                          years: str = 'VOC2012',
                          source_dict: dict = None,
                          objects_list: list = None,
                          im_shape: tuple = None,
                          save_root: str = os.getcwd(),
                          cover: bool = False):
    if not (filename and source_dict and objects_list and im_shape):
        return

    # 0--Parade/0_Parade_marchingband_1_849.jpg -> 0_Parade_marchingband_1_849.xml
    xml_name = filename.split(os.sep)[-1].split(".")[0] + '.xml'
    xml_full_path = os.path.join(save_root, xml_name)
    if os.path.exists(xml_full_path) and (cover is False):
        print(f"{xml_full_path} already exist, skip.")
        return

    xml_generator = XMLGenerator(xml_full_path)

    # xml root node
    node_root = xml_generator.create_append_node('annotation')
    xml_generator.create_text_node(node_name='folder', node_value=years, root_node=node_root)
    xml_generator.create_text_node(node_name='filename', node_value=filename, root_node=node_root)

    # source
    node_source = xml_generator.create_append_node('source', root_node=node_root)
    xml_generator.create_text_node(node_name='database', node_value=source_dict['database'], root_node=node_source)
    xml_generator.create_text_node(node_name='annotation', node_value=source_dict['annotation'], root_node=node_source)
    xml_generator.create_text_node(node_name='image', node_value=source_dict['image'], root_node=node_source)

    # size
    node_size = xml_generator.create_append_node('size', root_node=node_root)
    xml_generator.create_text_node(node_name='height', node_value=str(im_shape[0]), root_node=node_size)
    xml_generator.create_text_node(node_name='width', node_value=str(im_shape[1]), root_node=node_size)
    xml_generator.create_text_node(node_name='depth', node_value=str(im_shape[2]), root_node=node_size)

    # segmented
    xml_generator.create_text_node(node_name='segmented', node_value='0', root_node=node_root)

    # object
    for i, ob in enumerate(objects_list):
        xml_generator.create_object_node(info_dict=ob, root_node=node_root)

    # XML write
    xml_generator.save_xml()


# def create_xml_test():
#     objects = []
#     ob = {'name': 'person', 'pose': 'Unspecified', 'truncated': '0', 'difficult': '0',
#           'xmin': '174', 'ymin': '101', 'xmax': '349', 'ymax': '351'}
#     objects.append(ob)
#     objects.append(copy.deepcopy(ob))
#
#     years = 'VOC2012'
#     filename = 'test.jpg'
#     source_dict = {'database': 'The VOC2007 Database', 'annotation': 'PASCAL VOC2007', 'image': 'flickr'}
#     im_width = '500'
#     im_height = '700'
#     im_depth = '3'
#     im_shape = (im_width, im_height, im_depth)
#     create_pascal_voc_xml(filename=filename, years=years,
#                           source_dict=source_dict, objects_list=objects,
#                           im_shape=im_shape)


def create_xml(labels: list, img_root: str, img_path: str, save_root: str) -> bool:
    source_dict = {'database': 'The WIDERFACE2017 Database',
                   'annotation': 'WIDERFACE 2017',
                   'image': 'WIDERFACE'}

    img_full_path = os.path.join(img_root, img_path)
    if os.path.exists(img_full_path):
        im = cv2.imread(img_full_path)
        im_shape = im.shape
    else:
        print(f"Warning: {img_path} does not exist, can't read image shape.")
        im_shape = (0, 0, 0)

    ob_list = []
    for ob in labels:
        if ob[7] == '1':
            # invalid face image, skip
            continue

        if int(ob[2]) <= 0 or int(ob[3]) <= 0:
            print(f"Warning: find bbox w or h <= 0, in {img_path}, skip.")
            continue

        ob_dict = {'name': 'face',
                   'truncated': '0' if ob[8] == '0' else '1',
                   'difficult': '1' if ob[4] == '2' or ob[8] == '2' else '0',
                   'xmin': ob[0], 'ymin': ob[1],
                   'xmax': str(int(ob[0]) + int(ob[2])),
                   'ymax': str(int(ob[1]) + int(ob[3])),
                   'blur': ob[4], 'expression': ob[5],
                   'illumination': ob[6], 'invalid': ob[7],
                   'occlusion': ob[8], 'pose': ob[9]}

        # if ob[7] == '1':
        #     cv2.rectangle(im, (int(ob_dict['xmin']), int(ob_dict['ymin'])),
        #                   (int(ob_dict['xmax']), int(ob_dict['ymax'])),
        #                   (0, 0, 255))
        #     cv2.imshow("s", im)
        #     cv2.waitKey(0)

        ob_list.append(ob_dict)

    if len(ob_list) == 0:
        print(f"in {img_path}, no object, skip.")
        return False

    create_pascal_voc_xml(filename=img_path,
                          years="WIDERFACE2017",
                          source_dict=source_dict,
                          objects_list=ob_list,
                          im_shape=im_shape,
                          save_root=save_root)

    return True


def parse_wider_txt(data_root: str, split: str, save_root: str):
    """
    refer to: torchvision.dataset.widerface.py
    :param data_root:
    :param split:
    :param save_root:
    :return:
    """
    assert split in ['train', 'val'], f"split must be in ['train', 'val'], got {split}"

    if os.path.exists(save_root) is False:
        os.makedirs(save_root)

    txt_path = os.path.join(data_root, 'wider_face_split', f'wider_face_{split}_bbx_gt.txt')
    img_root = os.path.join(data_root, f'WIDER_{split}', 'images')
    with open(txt_path, "r") as f:
        lines = f.readlines()
        file_name_line, num_boxes_line, box_annotation_line = True, False, False
        num_boxes, box_counter, idx = 0, 0, 0
        labels = []
        xml_list = []
        progress_bar = tqdm(lines)
        for line in progress_bar:
            line = line.rstrip()
            if file_name_line:
                img_path = line
                file_name_line = False
                num_boxes_line = True
            elif num_boxes_line:
                num_boxes = int(line)
                num_boxes_line = False
                box_annotation_line = True
            elif box_annotation_line:
                box_counter += 1
                line_split = line.split(" ")
                line_values = [x for x in line_split]
                labels.append(line_values)
                if box_counter >= num_boxes:
                    box_annotation_line = False
                    file_name_line = True

                    if num_boxes == 0:
                        print(f"in {img_path}, no object, skip.")
                    else:
                        if create_xml(labels, img_root, img_path, save_root):
                            # 只记录有目标的xml文件
                            xml_list.append(img_path.split("/")[-1].split(".")[0])

                    box_counter = 0
                    labels.clear()
                    idx += 1
                    progress_bar.set_description(f"{idx} images")
            else:
                raise RuntimeError("Error parsing annotation file {}".format(txt_path))

        with open(split + '.txt', 'w') as w:
            w.write("\n".join(xml_list))


def gen_train_file(root: str):
    """
    <object>
        <bndbox>
            <xmin>495</xmin>
            <ymin>177</ymin>
            <xmax>532</xmax>
            <ymax>228</ymax>
        </bndbox>
        <name>face</name>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <blur>0</blur>
        <expression>0</expression>
        <illumination>0</illumination>
        <invalid>0</invalid>
        <occlusion>0</occlusion>
        <pose>0</pose>
    </object>
    """
    trainval_file = os.path.join(root, 'ImageSets/Main/trainval.txt')
    xml_dir = os.path.join(root, 'Annotations')

    for xml_file in os.listdir(xml_dir):
        file_name, file_tyep = os.path.splitext(xml_file)
        with open(trainval_file, 'a', encoding='UTF-8') as f:
            f.write(file_name + "\n")

        # xml_path = os.path.join(root, xml_file)
        #
        # anno = ET.parse(xml_file).getroot()  # 读取xml文档的根节点
        #
        # for obj in anno.iter("object"):
        #
        #     if int(obj.find("truncated").text) == 1:
        #         continue
        #
        #     if int(obj.find("difficult").text) == 1:
        #         continue
        #
        #     if int(obj.find("blur").text) == 2:
        #         continue
        #
        #     # if int(obj.find("expression").text) == 2:
        #     #     continue
        #
        #     if int(obj.find("illumination").text) == 1:
        #         continue
        #
        #     if int(obj.find("invalid").text) == 1:
        #         continue
        #
        #     # if int(obj.find("occlusion").text) == 1:
        #     #     continue
        #
        #     # if int(obj.find("pose").text) == 1:
        #     #     continue


if __name__ == '__main__':
    # 生成voc格式的标注文件
    # parse_wider_txt("/home/ubuntu/data/wider_face/",
    #                 "train",
    #                 "/home/ubuntu/data/wider_face/annotation/")

    gen_train_file("/home/ubuntu/data/wider_face")

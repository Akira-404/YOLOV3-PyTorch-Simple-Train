import os
import cv2


def main():
    root = '/home/ubuntu/data/mask'
    anno_path = os.path.join(root, 'Annotations')
    images_path = os.path.join(root, 'JPEGImages')
    anno_dir = os.listdir(anno_path)
    images_dir = os.listdir(images_path)
    print(f'len: {len(anno_dir)}')
    print(f'len: {len(images_dir)}')

    for xml_file in anno_dir:
        image_file = xml_file.replace('xml', 'jpg')
        image_file = os.path.join(root, 'JPEGImages', image_file)
        print(image_file)
        im = cv2.imread(image_file)
        cv2.imshow('image', im)
        cv2.waitKey(30)


# trainval_file = '/home/ubuntu/data/mask/ImageSets/Main/trainval.txt'
# for xml_file in anno_dir:
#     file_name, file_tyep = os.path.splitext(xml_file)
#     with open(trainval_file, 'a', encoding='UTF-8') as f:
#         f.write(file_name + "\n")


if __name__ == '__main__':
    main()

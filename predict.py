import os
import time
import argparse
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image
from utils.utils_prediect import Predict
from utils.utils import load_yaml_conf

parse = argparse.ArgumentParser('predict config')
parse.add_argument('-m', '--mode', type=str, choices=['image', 'video', 'dir', 'test'], default='image',
                   help='predict image or video or dir')
parse.add_argument('-i', '--image', type=str, default='person.jpeg',
                   help='image path')
parse.add_argument('-v', '--video', type=str, default='',
                   help='video path')
parse.add_argument('-d', '--dir', type=str,
                   default='/home/cv/PycharmProjects/rabbitmq-proj/download/onnxsim/cloud/2021915',
                   help='dir path')
parse.add_argument('-s', '--save_path', type=str, default='./out/person_spp')
args = parse.parse_args()

predict = Predict('predict.yaml')
predict.load_weights()
conf = load_yaml_conf('predict.yaml')
conf = conf['object'][conf['obj_type']]


# output_path = ''
# if os.path.exists(args.save_path) is False:
#     os.mkdir(args.save_path)


def run_test_image():
    test_file = os.path.join(conf['dataset_root'], 'VOCdevkit/VOC2007/ImageSets/Main/test.txt')
    with open(test_file, 'r') as f:
        lines = f.readlines()
    # for i, line in enumerate(lines):
    cnt = 0
    for line in tqdm(lines):
        line = line.strip()
        img_path = os.path.join(conf['dataset_root'], 'VOCdevkit/VOC2007/JPEGImages', line) + '.jpg'
        assert os.path.exists(img_path), f'{img_path} is error'
        # print(img_path)
        if img_path.lower().endswith('jpg'):
            image = Image.open(img_path)
            r_image = predict.detect_image(image)
            r_image.save(os.path.join(args.save_path, line) + '.jpg')
        cnt += 1
        if cnt > 100:
            break


def main(args):
    if args.mode == 'image':
        if args.image != "":
            if os.path.exists(args.image) is True:
                image = Image.open(args.image)
                ret_image, _ = predict.detect_image(image)
                ret_image.show()
            else:
                print(f'{args.image} is error')
    elif args.mode == 'video':

        assert os.path.exists(args.video) is True, f'{args.video} is error'
        _fps = 0.0
        _video_save_fps = 30.0
        _video_save_path = os.path.join(args.save_path, 'video.mp4')

        capture = cv2.VideoCapture(args.video)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(_video_save_path, fourcc, _video_save_fps, size)

        while True:
            t1 = time.time()
            ref, frame = capture.read()
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            frame, _ = np.array(predict.detect_image(frame))
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fps = (_fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % fps)
            frame = cv2.putText(frame, "fps= %.2f" % _fps, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if _video_save_path != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break
        capture.release()
        out.release()
        cv2.destroyAllWindows()
    elif args.mode == 'dir':
        assert os.path.exists(args.dir) is True, f'{args.dir} is error'
        _support_img_type = ['.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff']
        # _dir_out_path = os.path.join(args.save_path, os.path.basename(args.dir))

        image_names = os.listdir(args.dir)
        for image_name in tqdm(image_names):
            print(image_name)
            if image_name.lower().endswith('jpg'):
                image_path = os.path.join(args.dir, image_name)
                image = Image.open(image_path)
                r_image, _ = predict.detect_image(image)

                r_image.save(os.path.join(args.save_path, image_name))
    elif args.mode == 'test':
        run_test_image()
    else:
        raise TypeError('args.mode must be:image,video,dir')


if __name__ == '__main__':
    main(args)

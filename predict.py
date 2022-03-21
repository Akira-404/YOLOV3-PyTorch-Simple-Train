import os
import time
import argparse
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image
from utils.utils_prediect import Predict
from utils.utils import load_yaml_conf

predict = Predict('predict.yaml')
predict.load_weights()
conf = load_yaml_conf('predict.yaml')
conf = conf['object'][conf['obj_type']]


def main(args):
    if args.mode == 'image':
        if args.image != "":
            if os.path.exists(args.image) is True:
                image = Image.open(args.image)
                data = predict.detect_image(image)
                print(f'data:{data}')

                # PIL:Image->OpenCV
                frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

                for item in data:
                    x1, y1 = item['left'], item['top']
                    x2, y2 = item['left'] + item['width'], item['top'] + item['height']
                    cv2.rectangle(frame, (x1 - 10, y1 - 10), (x2 + 10, y2 + 10), (255, 0, 0), 1)
                    cv2.putText(frame, str(item['label']), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))
                    cv2.putText(frame, str(round(item['score'], 3)), (x1, y1 + 15), cv2.FONT_HERSHEY_PLAIN, 1,
                                (255, 0, 0))
                cv2.imshow("video", frame)
                cv2.waitKey(0)
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
            ref, frame = capture.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # OpenCV->PIL:Image
            image = Image.fromarray(np.uint8(frame))
            data = predict.detect_image(image)
            print(f'data:{data}')

            for item in data:
                x1, y1 = item['left'], item['top']
                x2, y2 = item['left'] + item['width'], item['top'] + item['height']
                cv2.rectangle(frame, (x1 - 10, y1 - 10), (x2 + 10, y2 + 10), (255, 0, 0), 1)
                cv2.putText(frame, str(item['label']), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))
                cv2.putText(frame, str(round(item['score'], 3)), (x1, y1 + 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))
            cv2.imshow("video", frame)

            c = cv2.waitKey(25) & 0xff
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
                data = predict.detect_image(image)
                print(data)
                image.show()
                # r_image.save(os.path.join(args.save_path, image_name))

    else:
        raise TypeError('args.mode must be:image,video,dir')


if __name__ == '__main__':
    parse = argparse.ArgumentParser('predict config')
    parse.add_argument('-m', '--mode', type=str, choices=['image', 'video', 'dir'], default='video',
                       help='predict image or video or dir')
    parse.add_argument('-i', '--image', type=str, default='./person.jpeg',
                       help='image path')
    parse.add_argument('-v', '--video', type=str, default='/home/ubuntu/github/opencv/samples/data/vtest.avi',
                       help='video path')
    parse.add_argument('-d', '--dir', type=str,
                       default='',
                       help='dir path')
    parse.add_argument('-s', '--save_path', type=str, default='')
    args = parse.parse_args()
    main(args)

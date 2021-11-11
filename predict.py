import os
import time
import argparse
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image
from utils.utils_prediect import Predict

parse = argparse.ArgumentParser('predict config')
parse.add_argument('-m', '--mode', type=str, choices=['image', 'video', 'dir'], default='image',
                   help='predicct image or video or dir')
parse.add_argument('-i', '--image', type=str, default='',
                   help='image path')
parse.add_argument('-v', '--video', type=str, default='',
                   help='video path')
parse.add_argument('-d', '-dir', type=str, default='',
                   help='dir path')
args = parse.parse_args()

predict = Predict('predict.yaml')

output_path = './out'


def main(args):
    if args.mode == 'image':
        if args.image != "":
            if os.path.exists(args.image) is True:
                image = Image.open(args.image)
                ret_image = predict.detect_image(image)
                ret_image.show()
            else:
                print(f'{args.image} is error')
    elif args.mode == 'video':

        assert os.path.exists(args.video) is True, f'{args.video} is error'
        _fps = 0.0
        _video_save_fps = 30.0
        _video_save_path = os.path.join(output_path, 'video.mp4')

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
            frame = np.array(predict.detect_image(frame))
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
        _dir_out_path = os.path.join(output_path, 'dir_out')

        image_names = os.listdir(args.dir_path)
        for image_name in tqdm(image_names):
            if image_name.lower().endswith(_support_img_type):
                image_path = os.path.join(args.dir, image_name)
                image = Image.open(image_path)
                r_image = predict.detect_image(image)

                if not os.path.exists(_dir_out_path):
                    os.makedirs(_dir_out_path)
                r_image.save(os.path.join(_dir_out_path, image_name))
    else:
        raise TypeError('args.mode must be:image,video,dir')


if __name__ == '__main__':
    main(args)

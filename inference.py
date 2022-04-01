# Based on YOLOv5 by Ultralytics (under GPL-3.0 license)
"""
Run inference on images, videos, directories, streams, etc.

Usage when testing (not using camera): sources:
    $ python path/to/inference.py --test --source   img.jpg        # image
                                                    vid.mp4        # video
                                                    path/          # directory
                                                    path/*.jpg     # glob
                                                    'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                    'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/inference.py --weights flir-yolov5.pt      # yolov5 (PyTorch)
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

import json
import requests

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_img_size, check_imshow, check_requirements,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync, prune

#defaults
endpoint = '172.24.121.16:8000'
weights = 'flir-yolov5.pt'
data = 'flir-yolov5.yaml'

ifx='negative', # negative effect to replicate thermal

@torch.no_grad()
def run(endpoint=endpoint, # endpoint for API
        weights=weights,  # model.pt path(s)
        data=data,  # dataset.yaml path
        test=False,
        multitest=False,
        pruning=0.25,
        source='images',  # file directory if running algorithm to test
        imgsz=(640, 512),  # needed inference size for model size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'tests',  # save results to project/name
        name='test',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    is_test = test or multitest
    camera = not is_test
    source = str(source)

    # Directories to save testing
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Pruning model
    if pruning:
        prune(model, pruning)

    # Dataloader
    if camera:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0

    run_data = []
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        
        json_dict = {} # json output

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if camera:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            if not multitest:
                s += '%gx%g ' % im.shape[2:]  # print string
            
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    
                    json_dict[f'{names[int(c)]}'] = f'{n}'

                # Write results
                for *xyxy, conf, cls in reversed(det):

                    if is_test or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if test:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        sec = f'{t3 - t2:.3f}'
        # Print time (inference-only)
        if not multitest:
            LOGGER.info(f'{s}Done. ({sec}s)')
        #LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

        # Send data to API endpoint
        data_json = json.dumps(json_dict)
        url = 'http://' + endpoint + '/update'
        headers = {
            "Content-Type": "application/json",
        }
        if not multitest:
            try:
                r = requests.post(url, data=data_json, headers=headers)
            except requests.exceptions.RequestException as err:
                raise SystemExit(err)

        run_data.append({'sec': sec, 'data': json_dict})        

    # Print results
    if not multitest:
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)
    
    return run_data


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint', type=str, default=endpoint, help='endpoint for API')
    parser.add_argument('--weights', nargs='+', type=str, default='flir-yolov5.pt', help='model path(s)')
    parser.add_argument('--test', action='store_true', help='if testing model, add True')
    parser.add_argument('--multitest', action='store_true', help='if multiple testing, add True')
    parser.add_argument('--pruning', type=float, default=0.25, help='level of pruning')
    parser.add_argument('--data', type=str, default='flir-yolov5.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--source', type=str, default=ROOT / 'images', help='(optional) source if not camera')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference ,camera')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'tests', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt

def multitest(opt):
    total_run_data = []
    number_of_runs = 2 #number of runs purely to test model's mAP and accuracy

    correct_person_numers = [7, 6, 6, 5, 4] #correct number in image 1, 2, 3, 4, and 5 [7, 6, 6, 5, 4]
    
    for i in range(number_of_runs):
        print(f'RUN {i+1}')
        run_data = run(**vars(opt))
        for j in range(len(run_data)):
            if run_data[j]['data']['person']:
                print('Speed: ' + run_data[j]['sec'] + 's; People found: ' + run_data[j]['data']['person'])
                diff = 0
                persons = int(run_data[j]['data']['person'])
                correct_persons = correct_person_numers[j]
                if persons > correct_persons:
                    diff = persons - correct_persons
                elif correct_persons > persons:
                    diff = correct_persons - persons
                run_data[j]['accuracy'] = (correct_persons - diff) / correct_persons
        print('RUN COMPLETE')
        print()
        total_run_data.append(run_data)

    print(total_run_data)
    total_accuracies = 0
    total_speeds = 0
    for i in range(len(total_run_data)):
        run_accuracies = 0
        run_speeds = 0
        for j in range(len(total_run_data[i])):
            run_accuracies += total_run_data[i][j]['accuracy']
            run_speeds += float(total_run_data[i][j]['sec'])
        total_accuracies += run_accuracies / len(total_run_data[i])
        total_speeds += run_speeds / len(total_run_data[i])
    mean_average_precision = total_accuracies / len(total_run_data)
    average_speed = total_speeds / len(total_run_data)
    print()
    print(f'Average speed: {average_speed:.3f}s')
    print(f'mAP: {mean_average_precision:.3f}')
    
def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    if opt.multitest:
        multitest(opt)
    else: 
        try:
            r = requests.get('http://' + endpoint)
            run(**vars(opt))
        except requests.exceptions.RequestException as err:
            raise SystemExit(err)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

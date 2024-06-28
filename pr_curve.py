import os
import sys
import mmcv
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
import copy
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from mmengine import Config
# from mmcv import Config
# from mmdet.datasets import build_dataset
import mmengine
from mmengine.dataset import ClassBalancedDataset, ConcatDataset
from mmdet.datasets.dataset_wrappers import MultiImageMixDataset
from mmdet.registry import DATASETS
from mmengine.registry import init_default_scope
# from mmdet.core import results2json
def convert_to_coco_json(pkl_results):
    json_results = []
    for result in pkl_results:
        image_id = result['img_id']
        for bbox, label, score in zip(result['pred_instances']['bboxes'],
                                      result['pred_instances']['labels'],
                                      result['pred_instances']['scores']):
            bbox = bbox.tolist()  # Convert tensor to list
            score = score.item()  # Convert tensor to float
            label = label.item() # Convert tensor to int and adjust label (if necessary)

            # Convert bbox from [x1, y1, x2, y2] to [x, y, width, height]
            bbox_coco = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

            json_results.append({
                "image_id": image_id,
                "category_id": label,
                "bbox": bbox_coco,
                "score": score
            })
    return json_results


def build_dataset(cfg, default_args=None):

    if cfg['type'] == 'ClassBalancedDataset':
        dataset = ClassBalancedDataset(
            build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
    elif cfg['type'] == 'MultiImageMixDataset':
        cp_cfg = copy.deepcopy(cfg)
        cp_cfg['dataset'] = build_dataset(cp_cfg['dataset'])
        cp_cfg.pop('type')
        dataset = MultiImageMixDataset(**cp_cfg)
    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        dataset = ConcatDataset(cfg, default_args)
    else:
        dataset = DATASETS.build(cfg, default_args=default_args)

    return dataset

def plot_pr_curve(config_file, result_file, metric="bbox"):
    """plot precison-recall curve based on testing results of pkl file.

        Args:
            config_file (list[list | tuple]): config file path.
            result_file (str): pkl file of testing results path.
            metric (str): Metrics to be evaluated. Options are
                'bbox', 'segm'.
    """
    #
    cfg = Config.fromfile(config_file)
    init_default_scope(cfg.get('default_scope', 'mmdet'))
    # # turn on test mode of dataset
    # if isinstance(cfg.data.test, dict):
    #     cfg.data.test.test_mode = True
    # elif isinstance(cfg.data.test, list):
    #     for ds_cfg in cfg.data.test:
    #         ds_cfg.test_mode = True


    # dataset = DATASETS.build(cfg.test_dataloader.dataset)
    # predictions = mmengine.load(args.pkl_results)
    #
    # evaluator = Evaluator(cfg.val_evaluator)
    # evaluator.dataset_meta = dataset.metainfo
    # eval_results = evaluator.offline_evaluate(predictions)

    # build dataset
    dataset = build_dataset(cfg.test_dataloader.dataset)
    # load result file in pkl format
    pkl_results = mmengine.load(result_file)
    # convert pkl file (list[list | tuple | ndarray]) to json
    # json_results, _ = dataset.format_results(pkl_results)

    # Example usage
    json_results = convert_to_coco_json(pkl_results)
    with open('results.json', 'w') as f:
        json.dump(json_results, f)


    # initialize COCO instance
    coco = COCO(annotation_file='data/SSDD_aug/test/annotation_coco.json')
    coco_gt = coco
    coco_dt = coco_gt.loadRes(json_results)
    # initialize COCOeval instance
    coco_eval = COCOeval(coco_gt, coco_dt, metric)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    # extract eval data
    precisions = coco_eval.eval["precision"]
    return precisions
    '''
    precisions[T, R, K, A, M]
    T: iou thresholds [0.5 : 0.05 : 0.95], idx from 0 to 9
    R: recall thresholds [0 : 0.01 : 1], idx from 0 to 100
    K: category, idx from 0 to ...
    A: area range, (all, small, medium, large), idx from 0 to 3
    M: max dets, (1, 10, 100), idx from 0 to 2
    '''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('config', help='config file path')
    # parser.add_argument('pkl_result_file', help='pkl result file path')
    parser.add_argument('--out', default='faster_rcnn_aug.png')
    parser.add_argument('--eval', default='bbox')
    cfg = parser.parse_args()

    out_pic = cfg.out
    lw = 0.5
    precisions2 = plot_pr_curve(config_file='work_dirs/faster_rcnn_ssdd/faster_rcnn_ssdd.py', result_file='20.pkl', metric=cfg.eval)

    pr_array1 = precisions2[0, :, 0, 0, 2]
    pr_array2 = precisions2[1, :, 0, 0, 2]
    pr_array3 = precisions2[2, :, 0, 0, 2]
    pr_array4 = precisions2[3, :, 0, 0, 2]
    pr_array5 = precisions2[4, :, 0, 0, 2]
    pr_array6 = precisions2[5, :, 0, 0, 2]
    pr_array7 = precisions2[6, :, 0, 0, 2]
    pr_array8 = precisions2[7, :, 0, 0, 2]
    pr_array9 = precisions2[8, :, 0, 0, 2]
    pr_array10 = precisions2[9, :, 0, 0, 2]

    x = np.arange(0.0, 1.01, 0.01)

    plt.plot(x, pr_array1, label="iou=0.5")
    plt.plot(x, pr_array2, label="iou=0.55")
    plt.plot(x, pr_array3, label="iou=0.6")
    plt.plot(x, pr_array4, label="iou=0.65")
    plt.plot(x, pr_array5, label="iou=0.7")
    plt.plot(x, pr_array6, label="iou=0.75")
    plt.plot(x, pr_array7, label="iou=0.8")
    plt.plot(x, pr_array8, label="iou=0.85")
    plt.plot(x, pr_array9, label="iou=0.9")
    plt.plot(x, pr_array10, label="iou=0.95")

    plt.xlabel("recall")
    plt.ylabel("precison")
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.01)
    plt.grid(True)
    plt.legend(loc="lower left")
    plt.savefig(out_pic)
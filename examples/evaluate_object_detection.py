import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from hocap_toolkit.utils import *


def evaluate_object_detection_results(gt_file, dt_file):
    coco_gt = COCO(str(gt_file))
    coco_dt = coco_gt.loadRes(str(dt_file))
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    ap_metrics = {
        "AP": coco_eval.stats[0],
        "AP50": coco_eval.stats[1],
        "AP75": coco_eval.stats[2],
        "APs": coco_eval.stats[3],
        "APm": coco_eval.stats[4],
        "APl": coco_eval.stats[5],
    }

    df = pd.DataFrame(ap_metrics)
    result_str = df.to_string(index=False)

    tqdm.write(result_str)

    # save to txt
    save_txt_file = dt_file.parent / f"{dt_file.stem}_AP.txt"
    save_txt_file.write_text(result_str)

    return ap_metrics


if __name__ == "__main__":
    benchmark_dir = PROJ_ROOT / "data/benchmarks"

    gt_file = PROJ_ROOT / "config/benchmarks/gt_object_detections_results.json"
    dt_file = PROJ_ROOT / "config/benchmarks/demo_object_detections_results.json"

    tqdm.write("- Evaluating Object Detection results...")

    ap_metrics = evaluate_object_detection_results(gt_file, dt_file)

    print(
        f"AP: {ap_metrics['AP']:.3f} | AP_50: {ap_metrics['AP50']:.3f} | AP_75: {ap_metrics['AP75']:.3f} | AP_s: {ap_metrics['APs']:.3f} | AP_m: {ap_metrics['APm']:.3f} | AP_l: {ap_metrics['APl']:.3f}"
    )

    tqdm.write("- Evaluation Done...")

from ast import dump
from suctionnetAPI import SuctionNetEval

if __name__ == "__main__":
    dataset_root = '/DATA2/Benchmark/suctionnet'
    camera = 'realsense'
    suctionnet_eval = SuctionNetEval(root=dataset_root, camera=camera)
    
    result_path = '/DATA2/Benchmark/suction/inference_results/deeplabV3plus'
    # evaluate all the test splits
    # res is the raw evaluation results, ap_top50 and ap_top1 are average precision of top 50 and top 1 suctions
    # see our paper for details
    res, ap_top50, ap_top1 = suctionnet_eval.eval_all(dump_folder=result_path, proc=30)

    # evaluation on a single split is also supported
    # evaluate seen split
    res, ap_top50, ap_top1 = suctionnet_eval.eval_seen(dump_folder=result_path, proc=30)
    # evaluate similar split
    res, ap_top50, ap_top1 = suctionnet_eval.eval_similar(dump_folder=result_path, proc=30)
    # evaluate novel split
    res, ap_top50, ap_top1 = suctionnet_eval.eval_novel(dump_folder=result_path, proc=30)

    # You may also save the evaluation results and get the precision whenever you want
    # res is the same with your saved results and ap is the average precision
    dump_file = 'result.npy'
    topk = 50   # 50 or 1
    res, ap = suctionnet_eval.get_precision_topk_all(dump_file, topk)

    # if you only save the evaluation results of a single split
    # you can use the following functions to get the precision
    # seen split
    res, ap = suctionnet_eval.get_precision_topk_seen(dump_file, topk)
    # similar split
    res, ap = suctionnet_eval.get_precision_topk_similar(dump_file, topk)
    # novel split
    res, ap = suctionnet_eval.get_precision_topk_novel(dump_file, topk)

from suctionnetAPI import SuctionNet


if __name__ == "__main__":
    dataset_root = '/DATA2/Benchmark/suctionnet'
    suctionnet = SuctionNet(root=dataset_root, camera='kinect')

    # visualize seal labels
    suctionnet.showObjSuction(obj_id=0, visu_num=10)
    # visualize collision labels
    suctionnet.showSceneCollision(scene_idx=0, anno_idx=0, camera='kinect', visu_num_each=10)
    # visualize wrench labels
    suctionnet.showSceneWrench(scene_idx=0, anno_idx=0, camera='kinect', visu_num_each=50)
    # only visualize scene
    suctionnet.show6DPose(scene_idx=0, anno_idx=0, camera='kinect')

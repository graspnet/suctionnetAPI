from suctionnetAPI import SuctionNet

if __name__ == "__main__":
    dataset_root = '/DATA2/Benchmark/graspnet'
    camera = 'realsense'
    suctionnet = SuctionNet(root=dataset_root, camera=camera)

    # check data completeness
    suctionnet.check_data_completeness()
    
    # we provide functions to explore the dataset
    # get a list of all the scenes that contain specific objects
    object_ids = [0, 1, 2]  # specify objects as you want
    scene_ids = suctionnet.getSceneIds(objIds=object_ids)

    # get a list of objects in specific scenes
    scene_ids = [0, 1, 2]   # specify scenes as you want
    object_ids = suctionnet.getObjIds(scene_ids)

    # we also provide functions to load data from the dataset
    # get object models (in the form of open3d.geometry.PointCloud)
    object_ids = [0, 1, 2]  # specify objects as you want
    model_list = suctionnet.loadObjModels(object_ids)

    # get object models (in the form of rimesh.Trimesh)
    object_ids = [0, 1, 2]  # specify objects as you want
    model_list = suctionnet.loadObjTrimesh(object_ids)

    # get a dict of seal labels of specific objects
    object_ids = [0, 1, 2]  # specify objects as you want
    seal_labels = suctionnet.loadSealLabels(object_ids)

    # get a dict of wrench labels of specific scenes
    scene_ids = [0, 1, 2]   # specify scenes as you want
    wrench_labels = suctionnet.loadWrenchLabels(scene_ids)

    # get a dict of collision labels of specific scenes
    scene_ids = [0, 1, 2]   # specify scenes as you want
    colli_labels = suctionnet.loadCollisionLabels(scene_ids)

    # get images
    # color image in RGB form
    rgb_img = suctionnet.loadRGB(sceneId=0, camera='realsense', annId=0)
    # color image in BGR form
    bgr_img = suctionnet.loadBGR(sceneId=0, camera='realsense', annId=0)
    # depth image
    depth_img = suctionnet.loadDepth(sceneId=0, camera='realsense', annId=0)
    # mask image
    mask_img = suctionnet.loadMask(sceneId=0, camera='realsense', annId=0)

    # get point clouds and corresponding colors from depth and color images
    # format can be open3d or numpy
    points, colors = suctionnet.loadScenePointCloud(sceneId=0, camera='realsense', annId=0, align=False, format = 'open3d')

    # get open3d point cloud models of a scene
    model_list = suctionnet.loadSceneModel(sceneId=0, camera ='kinect', annId = 0, align = False)

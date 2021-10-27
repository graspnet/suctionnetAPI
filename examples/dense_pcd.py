from suctionnetAPI import create_dense_point_cloud


if __name__ == "__main__":
    model_root = '/DATA2/Benchmark/graspnet/models'
    save_root = '/DATA2/Benchmark/suction/dense_pcd_test'
    create_dense_point_cloud(model_root, save_root)

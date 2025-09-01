# MVTec AD dataset
gpu=0
categories=("bottle" "cable" "capsule" "carpet" "grid" "hazelnut" "leather" "metal_nut" "pill" "screw" "tile" "toothbrush" "transistor" "wood" "zipper")
# input data path
dataset_path="data/mvtec_anomaly_detection"
anomaly_map_path="data/mvtec_musc_anomaly_map"
base_data_path="data/AeBAD_crop"
# output data path
binary_data_path="data/mvtec_musc" # the MEBin's output path of the binary data
crop_data_path="data/mvtec_musc_crop" # the MEBin's output path of the crop data
# experiment name
run_exp="mvtec_musc_crop"
checkpoint_dir="/home/xrli/datasets/Code/AnomalyNCD-main/outputs/mvtec_musc_crop/log"
for category in "${categories[@]}"; do
    find "$checkpoint_dir" -type d -name "*${category}*" | while read -r checkpoint_path; do
        CUDA_VISIBLE_DEVICES=$gpu python examples/anomalyncd_main.py \
            --runner_name "$run_exp" \
            --dataset "mvtec" \
            --category "$category" \
            --dataset_path "$dataset_path" \
            --anomaly_map_path "$anomaly_map_path" \
            --binary_data_path "$binary_data_path" \
            --crop_data_path "$crop_data_path" \
            --base_data_path "$base_data_path" \
            --only_test True \
            --checkpoint_path "$checkpoint_path"
    done
done


# # MTD dataset 
# gpu=0
# categories=("MTD")
# # input data path
# dataset_path="data/mtd_anomaly_detection/test"
# anomaly_map_path="data/mtd_musc_anomaly_map"
# base_data_path="data/AeBAD_crop"
# # output data path
# binary_data_path="data/mtd_musc"
# crop_data_path="data/mtd_musc_crop"
# run_exp="mtd_musc_crop"
# checkpoint_path="/home/xrli/datasets/Code/AnomalyNCD-main/outputs/mtd_musc_crop/log/AnomalyNCD_mtd_xxx"
# for category in "${categories[@]}"
# do
#     CUDA_VISIBLE_DEVICES=$gpu python examples/anomalyncd_main.py \
#         --runner_name "$run_exp" \
#         --dataset "mtd" \
#         --category "$category" \
#         --dataset_path "$dataset_path" \
#         --anomaly_map_path "$anomaly_map_path" \
#         --binary_data_path "$binary_data_path" \
#         --crop_data_path "$crop_data_path" \
#         --base_data_path "$base_data_path" \
#         --only_test True \
#         --checkpoint_path "$checkpoint_path"
# done
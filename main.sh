LEARNING_RATES=(5e-5) 
DROPOUT_RATES=(0.26)
BATCH_SIZES=(32)
NUMBLOCK=(2)
GSMAP_TIME_STEPS=(7)
ECMWF_TIME_STEPS=(7)
PATCH=(3)
SEEDS=(52) 
# Set name parameter by one of the following values: vifos-conv3d, vifos-spatial-temporal-extractor, cnn-lstm, unet, quantitle
for gsmap in "${GSMAP_TIME_STEPS[@]}"; do
  for ecmwf in "${ECMWF_TIME_STEPS[@]}"; do
    for bs in "${BATCH_SIZES[@]}"; do
      for dr in "${DROPOUT_RATES[@]}"; do
        for lr in "${LEARNING_RATES[@]}"; do
          for block in "${NUMBLOCK[@]}"; do
            for pat in "${PATCH[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    CUDA_VISIBLE_DEVICES=0 python main.py --cfg config/default.yaml \
                    --name vifos-conv3d \
                    --seed "$seed" \
                    --gsmap_time_step "$gsmap"\
                    --ecmwf_time_step "$ecmwf"\
                    --in_channel 13 \
                    --adding_type 0 \
                    --dropout "$dr" \
                    --height 25 \
                    --width 25 \
                    --data_idx_dir "<YOUR_DATA_IDX_DIR>" \
                    --gauge_data_path "<YOUR_GAUGE_DATA_PATH>" \
                    --npyarr_dir "<YOUR_NPYARR_DIR>" \
                    --processed_ecmwf_dir "<YOUR_PROCESSED_ECMWF_DIR>" \
                    --esp_data_path "<YOUR_GSMAP_DATA_PATH>"\
                    --lat_start 23.25 \
                    --lon_start 102.25 \
                    --height_esp 30 \
                    --width_esp 30 \
                    --lat_esp_start 23.25 \
                    --lon_esp_start 102.25 \
                    --use_layer_norm \
                    --loss_func weightedmse \
                    --lr "$lr" \
                    --use_lrscheduler \
                    --scheduler_type ReduceLROnPlateau \
                    --plateau_patience 3 \
                    --plateau_min_lr 1e-9 \
                    --plateau_factor 0.5 --plateau_verbose \
                    --num_vit_blocks "$block" \
                    --group_name data3-r1-test-vit-tiny-all-weekly \
                    --batch_size "$bs" \
                    --num_epochs 1000\
                    --patch_size "$pat" \
                    --output_norm \
                    --debug \

                done
            done
          done
        done
      done
    done
  done
done
        
        
        
        
        
        
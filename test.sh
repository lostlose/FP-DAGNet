python test_dvs.py \
--experiment_description 'test' \
--frame_method sbt \
-tf 20 -fs 3 -t 3 \
--batch_size 64 \
--img_size 256 \
--device 2 \
--fea_num_layers 10 \
--net_arch_fea='[0,0,1,1,1,2,2,2,3,3]' \
--cell_arch_fea='[[1, 1],[0, 1],[3, 2],[2, 1],[7, 1],[8, 1]]' \
--fea_filter_multiplier 32 \
--multi_anchor \
--conf_thresh 0.3 \
--weight "./weights/best.pth"
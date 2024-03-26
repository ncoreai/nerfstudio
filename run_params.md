# Parameters Used to 

### train the helical 
    
ns-train instant-ngp-bounded \
--load-dir /workspaces/nerfstudio/outputs/-workspaces-nerfstudio-data-dataset_hel_4k_train/instant-ngp-bounded/2023-03-01_021655/nerfstudio_models \
--viewer.skip-openrelay True \
--viewer.websocket-port 7008 \
--viewer.start-train False \
--viewer.max-num-display-images 100 \
--pipeline.datamanager.train-num-rays-per-batch 65536 \
--pipeline.model.cone-angle 0.0 \
--pipeline.model.grid-resolution 256 \
--machine.num-gpus 1 \
instant-ngp-data \
--data /workspaces/nerfstudio/data/dataset_hel_4k_train \
--scene-scale 2.0

### make helical render
``` bash
ns-render \
--load-config outputs/-workspaces-nerfstudio-data-dataset_hel_4k_train/instant-ngp-bounded/2023-03-05_172157/config.yml \
--traj filename \
--camera-path-filename /workspaces/nerfstudio/data/dataset_hel_4k_train/camera_paths/2023-03-05_172157.json \
--output-path renders/dataset_hel_4k_train/hel_vid__depth_.mp4 \
--rendered-output-names depth
```

```bash
ns-render --load-config outputs/-workspaces-nerfstudio-data-dataset_hel_4k_train/instant-ngp-bounded/2023-03-05_172157/config.yml --traj filename --camera-path-filename /workspaces/nerfstudio/data/dataset_hel_4k_train/camera_paths/2023-03-05_172157.json --output-path renders/dataset_hel_4k_train/2023-03-05_172157.mp4
```


### make helical orbit to compare 
``` bash
ns-render \
--load-config /workspaces/nerfstudio/outputs/-workspaces-nerfstudio-data-dataset_hel_4k_train/instant-ngp-bounded/2023-03-01_021655/config.yml \
--traj filename \
--camera-path-filename /workspaces/nerfstudio/data/dataset_flyby_4k_train/camera_paths/test_path.json \
--output-path renders/dataset_flyby_4k_train/hel_compare_flyby_depth_4.mp4 \
--rendered-output-names depth
```

## Flyby

### train the flyby
``` bash   
ns-train instant-ngp-bounded \
--load-dir /workspaces/nerfstudio/outputs/-workspaces-nerfstudio-data-dataset_flyby_4k_train/instant-ngp-bounded/2023-03-01_213857/nerfstudio_models \
--viewer.skip-openrelay True \
--viewer.websocket-port 7008 \
--viewer.start-train False \
--viewer.max-num-display-images 50 \
--pipeline.datamanager.train-num-rays-per-batch 65536 \
--pipeline.model.cone-angle 0.0 \
--pipeline.model.grid-resolution 128 \
--machine.num-gpus 1 \
instant-ngp-data \
--data /workspaces/nerfstudio/data/dataset_flyby_4k_train \
--scene-scale 2.0
```
``` bash
ns-render \
--load-config outputs/-workspaces-nerfstudio-data-dataset_flyby_4k_train/instant-ngp-bounded/2023-03-01_014552/config.yml \
--traj filename \
--camera-path-filename /workspaces/nerfstudio/data/dataset_flyby_4k_train/camera_paths/conpare_flyby.json \
--output-path renders/dataset_flyby_4k_train/conpare_flyby.mp4
```

``` bash
ns-render \
--load-config outputs/-workspaces-nerfstudio-data-dataset_flyby_4k_train/instant-ngp-bounded/2023-03-02_211158/config.yml \
--traj filename \
--camera-path-filename /workspaces/nerfstudio/data/dataset_flyby_4k_train/camera_paths/test_path.json \
--output-path renders/dataset_flyby_4k_train/compare_flyby_4.mp4
```
[flyby_compare_out]: renders/dataset_flyby_4k_train/compare_flyby_3.mp4


--pipeline.model.sdf-field.features-per-level INT  
--pipeline.model.sdf-field.log2-hashmap-size INT  
ns-train neus-facto --vis viewer+tensorboard --optimizers.fields.optimizer.lr 5e-5 --pipeline.model.eikonal-loss-mult 0.01 --optimizers.proposal-networks.optimizer.lr 0.0025 --pipeline.model.sdf-field.use-grid-feature True --pipeline.model.sdf-field.base-res 64 --pipeline.model.enable-collider True --pipeline.model.sdf-field.features-per-level 8 --pipeline.model.sdf-field.log2-hashmap-size 21 --pipeline.model.loss-coefficients.rgb-loss-coarse 2.0 --pipeline.model.loss-coefficients.rgb-loss-fine 2.0 --pipeline.model.background-model none --pipeline.model.sdf-field.bias 0.5 --pipeline.model.far-plane-bg 10.0 --pipeline.model.sdf-field.inside-outside False --pipeline.model.background-color black neus-data --data /nerf-sessions/nerf_data/ --train-split-fraction 0.9999  --auto-scale-poses True --scene_scale 1.0

ns-train neus-facto --vis viewer+tensorboard  --max-num-iterations 100001 --pipeline.model.num-proposal-iterations 3  --optimizers.proposal-networks.scheduler.milestones 15000 30000 50000 --optimizers.proposal-networks.scheduler.max-steps 100001 --pipeline.model.sdf-field.use-grid-feature True --pipeline.model.sdf-field.base-res 64 --pipeline.model.enable-collider True --pipeline.model.sdf-field.features-per-level 8 --pipeline.model.sdf-field.log2-hashmap-size 21 --pipeline.model.background-model none --pipeline.model.sdf-field.bias 0.5 --pipeline.model.far-plane-bg 10.0 --pipeline.model.sdf-field.inside-outside False --pipeline.model.background-color black neus-data --data /nerf-sessions/nerf_data/ --train-split-fraction 0.9999  --auto-scale-poses True --scene_scale 1.0
python3 classification.py -m 1.xml  
python3 object_detection.py -m 1.xml  

ssd_mobilenet_v1  
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --transformations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_support.json --input_model frozen_inference_graph.pb  --tensorflow_object_detection_api_pipeline_config pipeline.config

ssd_mobilenet_v2、ssd_incept_v2、ssd_resnet50  
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --transformations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --input_model frozen_inference_graph.pb  --tensorflow_object_detection_api_pipeline_config pipeline.config  

TX2  
head -n 1 /etc/nv_tegra_release  
# R32 (release), REVISION: 2.0, GCID: 15925399, BOARD: t186ref, EABI: aarch64, DATE: Sat Jul 13 07:31:45 UTC 2019  
JetPack 4.2.1 
L4T R32.2 (K4.9)
Ubuntu 18.04 LTS aarch64
CUDA 10.0.326
cuDNN 7.5.0.66
TensorRT 5.1.6.1
VisionWorks 1.6
OpenCV 3.3.1
Nsight Systems 2019.4
Nsight Graphics 2019.2
SDK Manager 0.9.13
sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v42 tensorflow-gpu==1.14.0+nv19.10  

python3 classification.py -m 1.xml  
python3 object_detection.py -m 1.xml

python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --transformations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --input_model ssd_mobilenet_v2_coco.frozen.pb  --tensorflow_object_detection_api_pipeline_config pipeline.config

import os
import sys
import cv2
import numpy as np
import logging as log
from argparse import ArgumentParser, SUPPRESS
from openvino.inference_engine import IENetwork, IECore
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.", required=True,type=str)
    args.add_argument("-i", "--input", help="Path to an image/video file.Default value is cam", default="cam", type=str)
    args.add_argument("-l", "--cpu_extension",help="Optional. Required for CPU custom layers. ", type=str, default=None)
    args.add_argument("-d", "--device",help="CPU, GPU, FPGA, HDDL, MYRIAD or HETERO .Default value is CPU",default="CPU", type=str)
    args.add_argument("-lb", "--labels", help="Optional. Path to a labels mapping file", default=None, type=str)

    return parser

def getmodel(model_xml,model_bin,device,cpu_extension,labels,log):
    # Plugin initialization for specified device and load extensions library if specified
    log.info("Creating Inference Engine")
    ie = IECore()
    if cpu_extension and 'CPU' in device:
        ie.add_extension(cpu_extension, "CPU")
    # Read IR
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)

    if "CPU" in device:
        supported_layers = ie.query_network(net, "CPU")
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)

    assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"

    log.info("Preparing input blobs")
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    net.batch_size = 1

    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=device)
    if labels:
        with open(labels, 'r') as f:
            labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]
    else:
        labels_map = None
    return ie,net,input_blob,out_blob,exec_net,labels_map
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    model_labels = os.path.splitext(model_xml)[0] + ".txt"
    labels = args.labels if args.labels else (model_labels if os.path.exists(model_labels) else None )
    ie,net,input_blob,out_blob,exec_net,labels_map = getmodel(model_xml,model_bin,args.device,args.cpu_extension,labels,log)
    Nn, Nc, Nh, Nw = net.inputs[input_blob].shape
 
    input_stream = 0 if args.input == "cam" else args.input
    cap = cv2.VideoCapture(input_stream)
    lastkey=0

    while cap.isOpened() and lastkey!=27 :
        ret, frame = cap.read()
        if not ret: break
        img =cv2.resize(frame, (Nw, Nh))
        if Nc==1: img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY ).reshape(Nh, Nw, 1)
        img = img.transpose((2, 0, 1)).reshape(Nn, Nc, Nh, Nw)
        res = exec_net.infer(inputs={input_blob: img})[out_blob][0]
        probs = np.squeeze(res)
        No1 = np.argsort(probs)[::-1][0]
        label = labels_map[No1] if labels_map else '#{}'.format(No1)

        cv2.putText(frame, '{}:{:.2f}%'.format(label, probs[No1]*100), (10, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow('Inference', frame)
        lastframe = frame.copy()
        lastkey = cv2.waitKey(1)

    cv2.imwrite("prediction.jpg",lastframe)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

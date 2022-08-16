
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
from util import * 
def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_
def parse_cfg_file(cfgfile):
    """
    Param: Configuration file

    Returns a list of blocks for network block configurations from the config file.
    Blocks are represented through dictionaries
    """
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')                        # store the lines in a list
    lines = [x for x in lines if len(x) > 0]               # get read of the empty lines 
    lines = [x for x in lines if x[0] != '#']              # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]           # get rid of fringe whitespaces
    
    block = {}
    blocks = []
    for line in lines:
        if line[0] == '[':
            blocks.append(block)
            block = {} # Start new dict
            block['type'] = line[1:-1].rstrip()
        else:
            key,value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks = blocks[1:]
    return blocks

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
class yolov3(nn.Module):
    def __init__(self, blocks) -> None:
        super(yolov3, self).__init__()
        self.blocks = blocks
        self.net_metadata = blocks[0]
        self.module_list = nn.ModuleList()
        self.prev_filter_depth = 3 # Starting with RGB
        self.output_filter_depth = []

        for idx, block in enumerate(blocks[1:]):
            module = nn.Sequential()
            
            # Check for conv layer
            if(block['type'] == 'convolutional'):
                activation = block['activation']

                if 'batch_normalize' in block:
                    batch_norm = True
                    bias = False
                else:
                    batch_norm = False
                    bias = True

                filters= int(block["filters"])
                padding = int(block["pad"])
                kernel_size = int(block["size"])
                stride = int(block["stride"])

                if padding:
                    pad = (kernel_size - 1) // 2
                else:
                    pad = 0
                
                #Add the convolutional layer
                conv = nn.Conv2d(self.prev_filter_depth, filters, kernel_size, stride, pad, bias = bias)
                module.add_module("conv_{0}".format(idx), conv)

                #Add the Batch Norm Layer
                if batch_norm:
                    bn = nn.BatchNorm2d(filters)
                    module.add_module("batch_norm_{0}".format(idx), bn)

            #Check the activation. 
            #It is either Linear or a Leaky ReLU for YOLO
                if activation == "leaky":
                    activn = nn.LeakyReLU(0.1, inplace = True)
                    module.add_module("leaky_{0}".format(idx), activn)

            #If it's an upsampling layer
            #We use Bilinear2dUpsampling
            elif (block["type"] == "upsample"):
                stride = int(block["stride"])
                upsample = nn.Upsample(scale_factor = 2, mode = "bilinear")
                module.add_module("upsample_{}".format(idx), upsample)

            #If it is a route layer
            elif (block["type"] == "route"):
                block["layers"] = block["layers"].split(',')
                #Start  of a route
                start = int(block["layers"][0])
                #end, if there exists one.
                try:
                    end = int(block["layers"][1])
                except:
                    end = 0
                #Positive anotation
                if start > 0: 
                    start = start - idx
                if end > 0:
                    end = end - idx
                route = EmptyLayer()
                module.add_module("route_{0}".format(idx), route)
                if end < 0:
                    filters = self.output_filter_depth[idx + start] + self.output_filter_depth[idx + end]
                else:
                    filters= self.output_filter_depth[idx + start]

            #shortcut corresponds to skip connection
            elif block["type"] == "shortcut":
                shortcut = EmptyLayer()
                module.add_module("shortcut_{}".format(idx), shortcut)
            
            #Yolo is the detection layer
            elif block["type"] == "yolo":
                mask = block["mask"].split(",")
                mask = [int(block) for block in mask]

                anchors = block["anchors"].split(",")
                anchors = [int(a) for a in anchors]
                anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
                anchors = [anchors[i] for i in mask]

                detection = DetectionLayer(anchors)
                module.add_module("Detection_{}".format(idx), detection)
            
            self.module_list.append(module)
            self.prev_filter_depth = filters
            self.output_filter_depth.append(filters)

    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}   #We cache the outputs for the route layer
        write = 0
        detections = None
        for idx, module in enumerate(modules):
            module_type = (module['type'])
            if module_type == 'convolutional' or module_type == 'upsample':
                x = self.module_list[idx](x)
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - idx

                if len(layers) == 1:
                    x = outputs[idx + (layers[0])]

                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - idx

                    map1 = outputs[idx + layers[0]]
                    map2 = outputs[idx + layers[1]]

                    x = torch.cat((map1, map2), 1)

            elif  module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[idx-1] + outputs[idx+from_]
            elif module_type == 'yolo':        

                anchors = self.module_list[idx][0].anchors
                #Get the input dimensions
                inp_dim = int (self.net_info["height"])

                #Get the number of classes
                num_classes = int (module["classes"])

                #Transform 
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:              #if no collector has been intialised. 
                    detections = x
                    write = 1

                else:       
                    detections = torch.cat((detections, x), 1)

            outputs[idx] = x
            return detections





blocks = parse_cfg_file('cfg/yolov3.cfg')
yolo_nw = yolov3(blocks)
model = yolo_nw
inp = get_test_input()
pred = model(inp, torch.cuda.is_available())
print (pred)            
import torch
from torch.autograd import Variable
import numpy as np
from scipy.special import softmax
from autoreconstruction.pytorch_segment.neurotorch.datasets.dataset import Data


class Predictor:
    """
    A predictor segments an input volume into an output volume
    """
    def __init__(self, net, checkpoint, gpu_device=None):
        gpu_device = gpu_device if torch.cuda.is_available() else None
        self.setNet(net, gpu_device=gpu_device)
        self.loadCheckpoint(checkpoint)

    def setNet(self, net, gpu_device=None):
        self.device = torch.device("cuda:{}".format(gpu_device)
                                   if gpu_device is not None
                                   else "cpu")

        self.net = net.to(self.device).eval()

    def getNet(self):
        return self.net

    def loadCheckpoint(self, checkpoint):
        self.getNet().load_state_dict(torch.load(checkpoint, map_location=self.device))

    def run(self, input_volume, output_volume, batch_size=100):
        self.setBatchSize(batch_size)

        with torch.no_grad():
            batch_list = [list(range(len(input_volume)))[i:i+self.getBatchSize()]
                          for i in range(0,
                                         len(input_volume),
                                         self.getBatchSize())]

            for batch_index in batch_list:
                batch = [input_volume[i] for i in batch_index]

                self.run_batch(batch, output_volume)

    def getBatchSize(self):
        return self.batch_size

    def setBatchSize(self, batch_size):
        self.batch_size = batch_size

    def run_batch(self, batch, output_volume):
        bounding_boxes, arrays = self.toTorch(batch)
        inputs = Variable(arrays).float()

        outputs = self.getNet()(inputs)
        
        tensor = torch.cat(outputs).data.cpu().numpy()
        tensor = np.uint8(np.round(255*softmax(tensor, axis=1)))  # Apply softmax and convert to uint8 before blend
        for ch in range(3):
            data_list = [Data(tensor[i][ch+1], bounding_box) for i, bounding_box in enumerate(bounding_boxes)]
            for data in data_list:
                output_volume[ch].blend(data)

    def toArray(self, data):
        torch_data = data.getArray().astype(float)
        torch_data = torch_data.reshape(1, 1, *torch_data.shape)
        return torch_data

    def toTorch(self, batch):
        bounding_boxes = [data.getBoundingBox() for data in batch]
        arrays = [self.toArray(data) for data in batch]
        arrays = torch.from_numpy(np.concatenate(arrays, axis=0))
        arrays = arrays.to(self.device)

        return bounding_boxes, arrays

    def toData(self, tensor_list, bounding_boxes):
        tensor = torch.cat(tensor_list).data.cpu().numpy()
        batch = [Data(tensor[i][0], bounding_box)
                 for i, bounding_box in enumerate(bounding_boxes)]

        return batch

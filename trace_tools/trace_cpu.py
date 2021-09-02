# -*- coding: utf-8 -*-
import torch.nn as nn
from mmseg.apis.inference import *


class Mmseg_deeplabv3plus(nn.Module):
    def __init__(self, model, img_metas):
        super(Mmseg_deeplabv3plus, self).__init__()
        self.model = model
        self.metas = img_metas
    def forward(self, tensor):
        with torch.no_grad():
            out = self.model(return_loss=False, rescale=False, img=[tensor], img_metas=self.metas)
        return out
         
if __name__ == '__main__':
    model_path = ''
    dst_trace_path = ''
    config_path = ''
    img_path = ''
    img = mmcv.imread(img_path)
    print('图片尺寸： ', img.shape)
    
    cfg = mmcv.Config.fromfile(config_path)
    device = 'cpu'  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    
    data = collate([data], samples_per_gpu=1)
    data['img_metas'] = [i.data[0] for i in data['img_metas']]
    data['img_metas'][0][0]['trace'] = 1
    model = init_segmentor(config_path, checkpoint=model_path, device='cpu')
    tensor = data['img'][0]
    print('图像增强后并转换为张量： ', tensor.size())
    
    model = Mmseg_deeplabv3plus(model, data['img_metas'])
    with torch.no_grad():
        print('.....start trace.....', end=' ')
        trace = torch.jit.trace(model, tensor)
        print('Finished !\n.....save pt model.....', end=' ')
        trace.save(dst_trace_path)
        print('Finished !')
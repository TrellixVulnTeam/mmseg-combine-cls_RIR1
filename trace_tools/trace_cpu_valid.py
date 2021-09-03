# -*- coding: utf-8 -*-
from mmseg.apis.inference import *
from collections import Counter 
from mmseg.ops import resize
import numpy as np
import os
if __name__ == '__main__':
    model_path = ''
    pytorch_model_path = ''
    config_path = ''
    img_path = ''
    output_path = ''
    cls_dict = {0: 'normal', 1: 'PPT'}
    #数据预处理
    img = mmcv.imread(img_path) #读取图片
    print(img.shape)
    cfg = mmcv.Config.fromfile(config_path) #编译cfg文件
    device = 'cpu'  #cpu
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:] #列表形式保存所有的预处理流程
    test_pipeline = Compose(test_pipeline) #将预处理流程合并为一个类
    data = dict(img=img) 
    data = test_pipeline(data) #预处理
    data = collate([data], samples_per_gpu=1) #预处理后数据提取
    data['img_metas'] = [i.data[0] for i in data['img_metas']]
    tensor = data['img'][0] #输入张量，这里len(data['img'])==1,即实际需要的tensor被放在一个列表中
    if device == 'cuda:0':
        tensor = tensor.cuda()
    print(tensor.size())

    # mmseg的pytorch模型， 用于检验pt模型是否与之一致
    mmsegmodel = init_segmentor(config_path, checkpoint=pytorch_model_path, device=device)
    # 工程用的c++ pt模型
    model = torch.jit.load(model_path, map_location=device)
    model.eval()

    with torch.no_grad():
        print('============================================================================\nbelow is pt model\'s seg result')
        seg_result, cls_result = model(tensor) #pt模型输出为一个四维张量(1, category+1, x, y), 1为数量，category为分割类别数
        print('img category: ', cls_dict[torch.max(cls_result, dim=1)[1].item()])
        #将张量的x, y变为图片原本尺寸
        seg_result = resize(
                seg_result,
                size=data['img_metas'][0][0]['ori_shape'][:2], #图片原本尺寸
                mode='bilinear', #线性插值填充值
                align_corners=False,
                warning=False)
        # 按维度1，也就是5 这个维度进行填入最大值的索引
        # 也就是(x, y)每一个像素点，都有5个通道，在5个通道上取最大值的索引值（经过softmax的score）
        # 即得到该像素点的分类
        seg_result = seg_result.argmax(dim=1).cpu().numpy()
        d = {0:0, 1:0, 2:0, 3:0, 4:0}
        for each in seg_result[0]:
            c = Counter(each)
            for eachk in c.keys():
                d[eachk] += c[eachk]
        print(d)
        #可视化
        seg = seg_result[0] #(x, y)
        palette = np.array([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
               [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
               [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
               [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
               [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]) #颜色
        opacity = 1  # 透明度，从0-1背景能见度越来越小
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) #(x, y, 3)
        #每一类的seg填充相应颜色
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        #seg与原图img合并
        color_seg = color_seg[..., ::-1]
        img = img * (1 - opacity) + color_seg * opacity
        img = img.astype(np.uint8)
        mmcv.imwrite(img, os.path.join(output_path,'pt_model_test.jpg')) #保存pt_model可视化


        print('============================================================================\nbelow is pytorch model\'s seg result')
        seg_result, cls_result = mmsegmodel(return_loss=False, rescale=True, img=[tensor], img_metas=data['img_metas'])
        print('img category: ', cls_dict[cls_result])
        #统计像素点分类情况，对比pt模型
        d = {0:0, 1:0, 2:0, 3:0, 4:0}
        for each in seg_result[0]:
            c = Counter(each)
            for eachk in c.keys():
                d[eachk] += c[eachk]
        print(d)
        #可视化
        mmsegmodel.show_result(mmcv.imread(img_path), seg_result, opacity=opacity, out_file=os.path.join(output_path,'pytorch_model_test.jpg'), palette=palette)

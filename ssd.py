import colorsys
import os
import time
import warnings
import torch.nn as nn
from nets.facenet import Facenet
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image, ImageDraw, ImageFont
from nets.ssd import SSD300
from utils.anchors import get_anchors
from utils.utils import cvtColor, get_classes, resize_image, preprocess_input, letterbox_image,compare_faces
from utils.utils_bbox import BBoxUtility
warnings.filterwarnings("ignore")
class SSD(object):
    _defaults = {

        #************************************************************************
        #*****************************SSD参数************************************
        #************************************************************************     
        #----------------------------------------------------------------------#
        #   U-MFF-SSD网络的权值,主干网络名称以及输入尺寸
        #----------------------------------------------------------------------#
        "model_path"        : "D:/OneDrive/python_work/SSD/训练wildface/wildface.pth",
        "classes_path"      : "E:/download/WildFace/Face_classes.txt",
        "input_shape"       : [300, 300],
        "backbone"          : "vgg",
        #----------------------------------------------------------------------#        
        #   目标检测置信度门限 及 非极大抑制所用到的参数
        #----------------------------------------------------------------------#
        "confidence"        : 0.5,
        "nms_iou"           : 0.45,
        #----------------------------------------------------------------------#   
        #   先验框大小,和特征图尺寸有关,在此处进行改动的话还需要在其他地方更改
        #----------------------------------------------------------------------#
        'anchors_size'      : [30, 60, 111, 162, 213, 264, 315],
        #----------------------------------------------------------------------#
        #   SSD网络是否进行不失真的resize
        #----------------------------------------------------------------------#
        "letterbox_image"   : True,



        #***********************************************************************
        #********************************facenet参数*****************************
        #***********************************************************************
        #----------------------------------------------------------------------#
        #   facenet网络的权值,主干网络名称以及输入尺寸
        "facenet_input_shape"   : [160, 160, 3],
        "facenet_backbone"      : "mobilenet",
        "facenet_model_path"    : "model_data/facenet_mobilenet.pth",
        #----------------------------------------------------------------------#
        #   facenet所使用的人脸距离门限
        #----------------------------------------------------------------------#
        "facenet_threhold"      : 0.9,
        #-------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------#
        "cuda"              : True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"


    #---------------------------------------------------#
    #   初始化ssd和facenet
    #---------------------------------------------------#
    def __init__(self, encoding =0,**kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        #---------------------------------------------------#
        #   计算总的类的数量
        #---------------------------------------------------#
        self.class_names, self.num_classes  = get_classes(self.classes_path)
        self.anchors                        = torch.from_numpy(get_anchors(self.input_shape, self.anchors_size, self.backbone)).type(torch.FloatTensor)
        if self.cuda:
            self.anchors = self.anchors.cuda()
        self.num_classes                    = self.num_classes + 1
        
        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.bbox_util = BBoxUtility(self.num_classes)

        #---------------------------------------------------#
        #   加载source人脸特征
        #---------------------------------------------------#       
        try:
            self.known_face_encodings = np.load("model_data/{backbone}_face_encoding.npy".format(backbone=self.facenet_backbone))
            self.known_face_names     = np.load("model_data/{backbone}_names.npy".format(backbone=self.facenet_backbone))
        except:
            if not encoding:
                print("载入已有人脸特征失败，请检查model_data下面是否生成了相关的人脸特征文件。")
            pass

        self.generate()

    #---------------------------------------------------#
    #   载入模型
    #---------------------------------------------------#
    def generate(self):
        #-------------------------------#
        #   载入SSD模型与权值
        #-------------------------------#
        self.net    = SSD300(self.num_classes, self.backbone)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))

        #-------------------------------#
        #   载入Facenet模型和权值
        #-------------------------------#   
        
        self.facenet    = Facenet(backbone=self.facenet_backbone, mode="predict").eval()    
        state_dict = torch.load(self.facenet_model_path)
        self.facenet.load_state_dict(state_dict, strict=False)


        #-------------------------------#
        #   模型加载到cuda上
        #-------------------------------#     
        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()

            self.facenet = nn.DataParallel(self.facenet)
            self.facenet = self.facenet.cuda()     
        print('Finished!')


    #   已知目标编码
    def encode_image(self,image,names):
        #获取图片的高和宽
        image_shape = np.array(np.shape(image)[0:2])
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        image       = cvtColor(image)
        #   给图像增加灰条，实现不失真的resize
        image_data  = resize_image(image, size= (self.input_shape[1], self.input_shape[0]), letterbox_image=True)
        #   添加上batch_size维度，图片预处理，归一化。
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            #   转化成torch的形式
            images = torch.from_numpy(image_data).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()
            #   将图像输入网络当中进行预测！
            outputs     = self.net(images)
            #   将预测结果进行解码
            results     = self.bbox_util.decode_box(outputs, self.anchors, image_shape, self.input_shape, self.letterbox_image, 
                                                    nms_iou = self.nms_iou, confidence = self.confidence)
            #   如果没有检测到物体，则返回原图
            if len(results[0]) <= 0:
                return image
            top_label   = np.array(results[0][:, 4], dtype = 'int32')   #用类别对应的序号标记类别
            top_conf    = results[0][:, 5]  #result 中的第五个位置的数字
            top_boxes   = results[0][:, :4] #result 中的0-4位置的数字
        if (len(top_label)==0):
            print('未检测到人脸')
            return
        elif (len(top_label)>1):
            print('检测到多张人脸,请重新输入')
            return
        #   进行目标的裁剪
        for i, c in list(enumerate(top_boxes)):
            top, left, bottom, right = top_boxes[i]
            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))
            
            dir_save_path = "img_crop"
            if not os.path.exists(dir_save_path):
                os.makedirs(dir_save_path)
            crop_image = image.crop([left, top, right, bottom])
            #   对人脸图片进行resize并改变通道顺序
            crop_img = np.array(letterbox_image(np.uint8(crop_image),(self.facenet_input_shape[1],self.facenet_input_shape[0])))/255
            crop_img = np.expand_dims(crop_img.transpose(2, 0, 1),0)
            #   利用facenet_model对resize后的参照人脸进行编码
            with torch.no_grad():
                crop_img = torch.from_numpy(crop_img).type(torch.FloatTensor)
                if self.cuda:
                    crop_img = crop_img.cuda()                
                face_encoding = self.facenet(crop_img)[0].cpu().numpy()
        return  face_encoding


    #   未知目标检测
    def detect_image(self, image):
        face_encodings = []
        #   计算输入图片的高和宽
        image_shape = np.array(np.shape(image)[0:2])
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        image       = cvtColor(image)
        #   给图像增加灰条，实现不失真的resize
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]),letterbox_image=True)
        #   添加上batch_size维度，图片预处理，归一化。
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        with torch.no_grad():
            #   转化成torch的形式
            images = torch.from_numpy(image_data).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()
            outputs     = self.net(images)
            #   将预测结果进行解码
            results     = self.bbox_util.decode_box(outputs, self.anchors, image_shape, self.input_shape, self.letterbox_image, 
                                                    nms_iou = self.nms_iou, confidence = self.confidence)
            #   如果没有检测到物体，则返回原图
            if len(results[0]) <= 0:
                return image
            #result中坐标为5(第6个)的数字,前四个是坐标,第五个是类别,第六个是置信度
            top_label   = np.array(results[0][:, 4], dtype = 'int32')   #坐标为4的数字(类别编号)
            top_conf    = results[0][:, 5]      #坐标为5的数字(置信度)
            top_boxes   = results[0][:, :4]     #坐标为0-3的数字,(坐标)
        #   设置字体与边框厚度
        font = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.input_shape[0], 1)
        
        #  裁剪出人脸
        for i, c in list(enumerate(top_boxes)):
            #人脸坐标
            top, left, bottom, right = top_boxes[i]
            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))    
            right   = min(image.size[0], np.floor(right).astype('int32'))
            # 剪裁图像
            dir_save_path = "img_crop"
            if not os.path.exists(dir_save_path):
                os.makedirs(dir_save_path)
            crop_image = image.crop([left, top, right, bottom])
            #展示剪裁之后的图片
            #crop_image.show()
            #保存剪裁之后的图片
            #crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
            #print("save crop_" + str(i) + ".png to " + dir_save_path)
            #   对人脸图片进行resize并改变通道顺序
            crop_img = np.array(letterbox_image(np.uint8(crop_image),(self.facenet_input_shape[1],self.facenet_input_shape[0])))/255
            crop_img = np.expand_dims(crop_img.transpose(2, 0, 1),0)
            with torch.no_grad():
                crop_img = torch.from_numpy(crop_img).type(torch.FloatTensor)
                if self.cuda:
                    crop_img = crop_img.cuda()                
            #   计算长度为128特征向量
                face_encoding = self.facenet(crop_img)[0].cpu().numpy()
                #print(face_encoding)
                #最终face_encodings为检测图片中检测到的所有人脸的编码
                face_encodings.append(face_encoding)
        
        face_names = []
        for face_encoding in face_encodings:
            #   取出一张脸并与数据库中所有的人脸进行对比，计算得分  
            matches, face_distances = compare_faces(self.known_face_encodings, face_encoding, tolerance = self.facenet_threhold)
            print(matches)
            print(face_distances)
            name = "Unknown"
            #   取出这个最近人脸的评分 取出当前输入进来的人脸，最接近的已知人脸的序号
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]: 
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        #   图像绘制
        for i, c in list(enumerate(top_label)):
            predicted_class = face_names[i]
            #print(predicted_class)
            #predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]
            top, left, bottom, right = box
            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))


            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
                 
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image

    def get_FPS(self, image, test_interval):
        #---------------------------------------------------#
        #   计算输入图片的高和宽
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]),letterbox_image=True)
        #---------------------------------------------------------#
        #   添加上batch_size维度，图片预处理，归一化。
        #---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            #---------------------------------------------------#
            #   转化成torch的形式
            #---------------------------------------------------#
            images = torch.from_numpy(image_data).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs     = self.net(images)
            #-----------------------------------------------------------#
            #   将预测结果进行解码
            #-----------------------------------------------------------#
            results     = self.bbox_util.decode_box(outputs, self.anchors, image_shape, self.input_shape, self.letterbox_image, 
                                                    nms_iou = self.nms_iou, confidence = self.confidence)

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                #---------------------------------------------------------#
                #   将图像输入网络当中进行预测！
                #---------------------------------------------------------#
                outputs     = self.net(images)
                #-----------------------------------------------------------#
                #   将预测结果进行解码
                #-----------------------------------------------------------#
                results     = self.bbox_util.decode_box(outputs, self.anchors, image_shape, self.input_shape, self.letterbox_image, 
                                                        nms_iou = self.nms_iou, confidence = self.confidence)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w") 
        #---------------------------------------------------#
        #   计算输入图片的高和宽
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), letterbox_image= True)
        #---------------------------------------------------------#
        #   添加上batch_size维度，图片预处理，归一化。
        #---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            #---------------------------------------------------#
            #   转化成torch的形式
            #---------------------------------------------------#
            images = torch.from_numpy(image_data).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs     = self.net(images)
            #-----------------------------------------------------------#
            #   将预测结果进行解码
            #-----------------------------------------------------------#
            results     = self.bbox_util.decode_box(outputs, self.anchors, image_shape, self.input_shape, self.letterbox_image, 
                                                    nms_iou = self.nms_iou, confidence = self.confidence)
            #--------------------------------------#
            #   如果没有检测到物体，则返回原图
            #--------------------------------------#
            if len(results[0]) <= 0:
                return 

            top_label   = np.array(results[0][:, 4], dtype = 'int32')
            top_conf    = results[0][:, 5]
            top_boxes   = results[0][:, :4]
        
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 

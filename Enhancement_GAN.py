#!/usr/bin/python3



#Libraries that we used in this python script
import os
import time
import numpy as np
import cv2
from scipy import ndimage
from util.parse_config import parse_config
import sys
import nibabel
import SimpleITK as sitk
import sys
from keras.models import model_from_json

# load GAN model here
json_file = open('/home/mohammad/pix2pixBig/models/generator_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into GAN model
loaded_model.load_weights("/home/mohammad/pix2pixBig/models/CNN36327/gen_weights_epoch395.h5")
print("/home/mohammad/pix2pixBig/models/CNN36327/gen_weights_epoch395.h5")


def resize_ND_volume_to_given_shape(volume, out_shape, order = 3):
    """
    resize an nd volume to a given shape
    inputs:
        volume: the input nd volume, an nd array
        out_shape: the desired output shape, a list
        order: the order of interpolation
    outputs:
        out_volume: the reized nd volume with given shape
    """
    shape0=volume.shape
    assert(len(shape0) == len(out_shape))
    scale = [(out_shape[i] + 0.0)/shape0[i] for i in range(len(shape0))]
    out_volume = ndimage.interpolation.zoom(volume, scale, order = order)
    return out_volume

def crop_ND_volume_with_bounding_box(volume, min_idx, max_idx):
    """
    crop/extract a subregion form an nd image.
    """
    dim = len(volume.shape)
    assert(dim >= 2 and dim <= 5)
    if(dim == 2):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1))]
    elif(dim == 3):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1))]
    elif(dim == 4):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1),
                               range(min_idx[3], max_idx[3] + 1))]
    elif(dim == 5):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1),
                               range(min_idx[3], max_idx[3] + 1),
                               range(min_idx[4], max_idx[4] + 1))]
    else:
        raise ValueError("the dimension number shoud be 2 to 5")
    return output

def get_ND_bounding_box(label, margin):
    """
    get the bounding box of the non-zero region of an ND volume
    """
    input_shape = label.shape
    if(type(margin) is int ):
        margin = [margin]*len(input_shape)
    assert(len(input_shape) == len(margin))
    indxes = np.nonzero(label)
    idx_min = []
    idx_max = []
    for i in range(len(input_shape)):
        idx_min.append(indxes[i].min())
        idx_max.append(indxes[i].max())

    for i in range(len(input_shape)):
        idx_min[i] = max(idx_min[i] - margin[i], 0)
        idx_max[i] = min(idx_max[i] + margin[i], input_shape[i] - 1)
    return idx_min, idx_max
    

def itensity_normalize_one_volume(volume):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzeor region
    inputs:
        volume: the input nd volume
    outputs:
        out: the normalized nd volume
    """
 
    pixels = volume[volume > 0]
    mean = pixels.mean()
    std  = pixels.std()
    out = (volume - mean)/std
    out_random = np.zeros(volume.shape)
    out[volume == 0] = out_random[volume == 0]
    return out


def itensity_normalize_black_pixel_volume(volume):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzeor region
    inputs:
        volume: the input nd volume
    outputs:
        out: the normalized nd volume
    """
    
    pixels = volume[volume > 0]
    #mean = pixels.mean()
    #std  = pixels.std()
    #out = (volume - mean)/std
    out_random = np.random.normal(0, 1, size = volume.shape)
    volume[volume == 0] = out_random[volume == 0]
    return volume    

    

def Enhancement_GAN(volume):
    imageF = itensity_normalize_one_volume(volume)
    #imageF = volume
    
    bbmin,bbmax = get_ND_bounding_box(imageF,0)
    
    volume=[]
    margin=0
    
    for zeroslice in range(0,bbmin[0]):
        recons_image=np.zeros((240,240))
        volume.append(recons_image) 
    
    for slices in range(bbmin[0],bbmax[0]):
        sliceimage=imageF[slices,:,:]
        
        bbslicemin,bbslicemax = get_ND_bounding_box(sliceimage, margin)
        output_shape=[128,128]
        in_center = [(bbslicemax[0]+bbslicemin[0])//2,(bbslicemax[1]+bbslicemin[1])//2]
        cropimage = crop_ND_volume_with_bounding_box(sliceimage, bbslicemin, bbslicemax)
        top = bbslicemin[0]
        bottom = 240-bbslicemax[0]-1
        left = bbslicemin[1]
        right=  240-bbslicemax[1]-1
        
        
        if(cropimage.shape[0]>128 or cropimage.shape[1]>128):
            resize_cropimage=resize_ND_volume_to_given_shape(cropimage, output_shape, order = 3)
            sub_data_moda= np.reshape(resize_cropimage,(1,resize_cropimage.shape[0],resize_cropimage.shape[1],1))
            if(sub_data_moda.shape==(1,128,128,1)):
                sub_data_moda=loaded_model.predict(sub_data_moda)
                GAN_IM=np.reshape(sub_data_moda,(sub_data_moda.shape[1],sub_data_moda.shape[2])) 

            
                original=resize_ND_volume_to_given_shape(GAN_IM, [cropimage.shape[0],cropimage.shape[1]], order = 3)
        
                recons_image = cv2.copyMakeBorder(original, top , bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0])
            else:
                
                recons_image =sliceimage
        else:
            cropimage = np.copy(sliceimage[in_center[0]-64:in_center[0]+64,in_center[1]-64:in_center[1]+64])

            sub_data_moda= np.reshape(cropimage,(1,cropimage.shape[0],cropimage.shape[1],1))
            if(sub_data_moda.shape==(1,128,128,1)):
                sub_data_moda=loaded_model.predict(sub_data_moda)
                GAN_IM=np.reshape(sub_data_moda,(sub_data_moda.shape[1],sub_data_moda.shape[2]))
            
                top = in_center[0]-64
                bottom = 240-64-in_center[0]
            
                left = in_center[1]-64
                right = 240-64-in_center[1]
                recons_image = cv2.copyMakeBorder(GAN_IM, top , bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0])
            else:
                recons_image =sliceimage######
        
        volume.append(recons_image)    
        
        
    for zeroslice in range(bbmax[0] ,155):
        recons_image=np.zeros((240,240))
        volume.append(recons_image)  
        
    volume=np.array(volume)  
    out_zero= np.zeros(imageF.shape)
    volume[imageF == 0] = out_zero[imageF == 0]
    volume=itensity_normalize_black_pixel_volume(volume) # For normalisation
    return volume
   
def load_one_volume(patient_name, mod,data_root,file_postfix):
        patient_dir = os.path.join(data_root, patient_name)
        # for bats17
        if('nii' in file_postfix):
            image_names = os.listdir(patient_dir)
            volume_name = None
            for image_name in image_names:
                if(mod + '.' in image_name):
                    volume_name = image_name
                    break
        else:
            img_file_dirs = os.listdir(patient_dir)
            volume_name  = None
            for img_file_dir in img_file_dirs:
                if(mod+'.' in img_file_dir):
                    volume_name = patient_dir + '/' + img_file_dir
                    break            
        
        volume_name = os.path.join(patient_dir, volume_name)
        volume = load_3d_volume_as_array(volume_name)
        
        return volume, volume_name    

def load_nifty_volume_as_array(filename, with_header = False):
    """
    load nifty image into numpy array, and transpose it based on the [z,y,x] axis order
    The output array shape is like [Depth, Height, Width]
    inputs:
        filename: the input file name, should be *.nii or *.nii.gz
        with_header: return affine and hearder infomation
    outputs:
        data: a numpy data array
    """
    img = nibabel.load(filename)
    data = img.get_data()
    data = np.transpose(data, [2,1,0])
    if(with_header):
        return data, img.affine, img.header
    else:
        return data 

def save_array_as_nifty_volume(data, filename, reference_name = None):
    """
    save a numpy array as nifty image
    inputs:
        data: a numpy array with shape [Depth, Height, Width]
        filename: the ouput file name
        reference_name: file name of the reference image of which affine and header are used
    outputs: None
    """
    img = sitk.GetImageFromArray(data)
    if(reference_name is not None):
        img_ref = sitk.ReadImage(reference_name)
        img.CopyInformation(img_ref)
    sitk.WriteImage(img, filename)


        
def load_3d_volume_as_array(filename):
    if('.nii' in filename):
        return load_nifty_volume_as_array(filename)
    elif('.mha' in filename):
        return load_mha_volume_as_array(filename)
    raise ValueError('{0:} unspported file format'.format(filename))   

def load_mha_volume_as_array(filename):
    img = sitk.ReadImage(filename)
    nda = sitk.GetArrayFromImage(img)
    return nda

def enhance(config_file):
    config = parse_config(config_file)
    config_data  = config['data']
    data_names   = config_data.get('data_names', None)
    data_root    = config_data.get('data_root', None)
    file_postfix    = config_data.get('file_postfix', None)
    
    modality_postfix     = config_data.get('modality_postfix', None)
    assert(os.path.isfile(data_names))
    with open(data_names) as f:
        content = f.readlines()
    patient_names = [x.strip() for x in content]
    data_num = len(patient_names)
    print(config_data)
    #data_num=2#################
    for i in range(data_num):
      volume_list = []
      volume_name_list = []
      for mod_idx in range(len(modality_postfix)):
        volume, volume_name = load_one_volume(patient_names[i], modality_postfix[mod_idx],data_root,file_postfix)
        
        start = time.clock()
        
        Enhanced_volume = Enhancement_GAN(volume)
        
        end = time.clock()
        print("Time per image: {} ".format((end-start))) 
        
        volume_name_enhanced = os.path.join(data_root, patient_names[i],patient_names[i][4:]+'_flairE4.nii.gz')
        
        save_array_as_nifty_volume(Enhanced_volume, volume_name_enhanced)
       
        print(volume_name_enhanced)
    
    #print(data_num)

if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print('Number of arguments should be 2. e.g.')
        print('python Enhancement_GAN.py config17/train_wt_ax.txt')
        sys.exit()#
    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))
    enhance(config_file)
    
    

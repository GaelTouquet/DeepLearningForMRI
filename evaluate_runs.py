import os
from NN.architectures import nrmse, nrmse_2D_L1,nrmse_2D_L2, get_wnet, get_unet
from NN.Generators import find_generator
from pandas import DataFrame
from tensorflow.keras.models import model_from_json
import tensorflow as tf

tf.config.set_visible_devices([],'GPU')

arch_dict = {    
    'trainingsaves_Unet_nonorm_img_Feb_03_13_13':get_unet(fullskip=False),
    'trainingsaves_Unet_nonorm_img_Feb_03_13_42':get_unet(fullskip=False),
    'trainingsaves_Unet_nonorm_img_skip_Feb_03_15_18':get_unet(fullskip=True),
    'trainingsaves_Unet_kspace_Feb_04_10_58':get_unet(fullskip=False),
    'trainingsaves_Unet_kspace_Feb_04_13_41':get_unet(fullskip=False),
    'trainingsaves_Unet_kspace_zeroinit_Feb_04_14_13':get_unet(fullskip=False),
    'trainingsaves_Unet_kspace_zeroinit_imgloss_Feb_04_16_01':get_unet(fullskip=False),
    'trainingsaves_Unet_img_zeroinit_absnormunnorm_Feb_05_11_14':get_unet(fullskip=True),
    'trainingsaves_Unet_img_absnormunnorm_Feb_05_11_42':get_unet(fullskip=True),
    'trainingsaves_Unet_img_absnormunnorm_depth8_Feb_05_12_56':get_unet(depth=8,fullskip=True),
    'trainingsaves_Unet_kspacetoimg_absnormunnorm_depth8_nobatchnorm_Feb_05_18_10':get_unet(depth=8,fullskip=True,batchnorm=False),
    'trainingsaves_Unet_img_absnormunnorm_depth8_nobatchnorm_Feb_05_13_54':get_unet(depth=8,fullskip=True,batchnorm=False),
    'trainingsaves_Unet_img_absnormunnorm_depth8_nobatchnorm_Feb_05_16_26':get_unet(depth=8,fullskip=True,batchnorm=False),
    'trainingsaves_Unet_img_absnormunnorm_depth8_nobatchnorm_nodropout_Feb_06_09_30':get_unet(depth=8,fullskip=True,batchnorm=False,dropout=0),
    # 'trainingsaves_Unet_kspacetoimg_absnormunnorm_depth9kernel1_nobatchnorm_Feb_08_15_15':get_unet(depth=9,kernel=1,n_filters=2,fullskip=True,batchnorm=False,dropout=0),
    # 'trainingsaves_Unet_kspacetoimg_absnormunnorm_depth9kernel2_nobatchnorm_Feb_08_17_28':get_unet(depth=9,kernel=2,fullskip=True,batchnorm=False,dropout=0),
}

base_workdir = r'D:\NN_DATA'

data_dirs = [f for f in os.listdir(base_workdir) 
if (os.path.isdir(os.path.join(base_workdir, f)) and ('legacy' not in f))]


categories = ['data','test type','path','nrmse L1','nrmse L2']#,'nrmse L1','nrmse L2','SSIM','path']
test_lists = []

for data_dir in data_dirs:
    test_dirs = [f for f in os.listdir(os.path.join(base_workdir,data_dir))
    if (os.path.isdir(os.path.join(base_workdir,data_dir,f))) and (f!='train') and (f!='val')]
    for test_dir in test_dirs:
        if not os.path.isfile(os.path.join(base_workdir,data_dir,test_dir,'epoch10.h5')) or test_dir == 'trainingsaves_Unet_kspacetoimg_absnormunnorm_depth9kernel1_nobatchnorm_Feb_08_15_15' or test_dir == 'trainingsaves_Unet_kspacetoimg_absnormunnorm_depth9kernel2_nobatchnorm_Feb_08_17_28':
            continue
        weight_path = os.path.join(base_workdir,data_dir,test_dir,'best.h5')
        if not os.path.isfile(weight_path):
            print('no weight file!')
            import pdb;pdb.set_trace()
        json_file = os.path.join(base_workdir,data_dir,test_dir,'model_save.json')
        if os.path.isfile(json_file):
            json_file = open(json_file, 'r')
            model = model_from_json(json_file.read())
            json_file.close()
        else:
            if test_dir in arch_dict:
                model = arch_dict[test_dir]
            else:
                print('Need model architecture!')
                import pdb;pdb.set_trace()
        try:
            model.load_weights(weight_path)
        except:
            continue
        model.compile(loss = nrmse_2D_L1,metrics=[nrmse_2D_L1,nrmse_2D_L2])
        generator = find_generator(os.path.join(base_workdir,data_dir,'val'))
        results = model.evaluate(generator)
        test_lists.append([
            data_dir,
            test_dir[14:-12],
            weight_path,
            results[1],
            results[2]
        ])

tests = DataFrame(test_lists,columns=categories)
tests.to_csv('test_results.csv')
print(tests)
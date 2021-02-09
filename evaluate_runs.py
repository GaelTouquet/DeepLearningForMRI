import os
from NN.architectures import nrmse, nrmse_2D_L1,nrmse_2D_L2, get_wnet
from NN.Generators import find_generator
from pandas import DataFrame
from tensorflow.keras.models import model_from_json
import tensorflow as tf

tf.config.set_visible_devices([],'GPU')

arch_dict = {
    'trainingsaves_Wnet_nonorm_img_Feb_03_13_13':get_wnet()
}

base_workdir = r'D:\NN_DATA'

data_dirs = [f for f in os.listdir(base_workdir) 
if (os.path.isdir(os.path.join(base_workdir, f)) and ('legacy' not in f))]


categories = ['data','test type','path']#,'nrmse L1','nrmse L2','SSIM','path']
test_lists = []

for data_dir in data_dirs:
    test_dirs = [f for f in os.listdir(os.path.join(base_workdir,data_dir))
    if (os.path.isdir(os.path.join(base_workdir,data_dir,f))) and (f!='train') and (f!='val')]
    for test_dir in test_dirs:
        if not os.path.isfile(os.path.join(base_workdir,data_dir,test_dir,'epoch10.h5')):
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
        import pdb;pdb.set_trace()
        model.load_weights(weight_path)
        model.compile(loss = nrmse_2D_L1,metrics=[nrmse, nrmse_2D_L1,nrmse_2D_L2])
        generator = find_generator(os.path.join(base_workdir,data_dir,'val'))
        results = model.evaluate(generator)
        import pdb;pdb.set_trace()
        test_lists.append([
            data_dir,
            test_dir[14:-12],
            weight_path
        ])

tests = DataFrame(test_lists,columns=categories)
print(tests)
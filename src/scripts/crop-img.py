import os,shutil
import cv2

source_root = 'I:/datasets/2d/expression recongition/ExpW/data/image/origin.7z/origin'
target_root = './ExpwCropped'
lst_file = 'G:/projects/pyProjects/netease-homework/expression-recongnition/data/datasets/label.lst'

if not os.path.exists(target_root):
    os.makedirs(target_root)


txt_count = 0
data_count = 0
miss_count = 0
with open(lst_file, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        
        txt_count+=1
        content = line.split(' ')
        meta_name = content[0]
        
        id_name = content[1]
        final_name = meta_name.split('.')[0] + '_'+str(id_name)+'.png'
        print(final_name)
        print(txt_count)
        src_path = os.path.join(source_root, meta_name)
        tgt_path = os.path.join(target_root,final_name)
#            path = os.path.join(img_path, content[0])
#            if os.path.exists(path) and int(content[7]) != 6:
        if os.path.exists(src_path):
            data_count+=1
            src_image = cv2.imread(src_path, 1)
            image = cv2.resize(src_image[int(content[2]):int(content[5]), int(content[3]):int(content[4])], (48, 48))
            cv2.imwrite(tgt_path,image)
#            data.append([path, int(content[7])])
        else:
            miss_count+=1

#
#a = []
#with open(lst_file, 'r', encoding='utf-8') as f:
#    for line in f.readlines():
#        content = line.split(' ')
#        meta_name = content[0]
#        id_name = content[1]
#        final_name = meta_name.split('.')[0] + '_'+str(id_name)+'.png'        
#        a.append(final_name)
#
#print(len(a))
#print(len(set(a)))
#
#imgs = os.listdir(source_root)
#for img in imgs:
#    if img.endswith('.png'):
#        print(img)
#        names = img.split('_')
#        l = len(names)
#        target_name = names[0]
#        
#        for i in range(1,l-1):
#            target_name += '_'
#            target_name += names[i]
#        
#        target_name += '.png'
#        shutil.copyfile(os.path.join(source_root,img),os.path.join(target_root,target_name))
        
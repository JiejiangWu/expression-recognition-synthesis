import os,shutil

source_root = './ExpwCleaned'
target_root = './ExpwCropped'

if not os.path.exists(target_root):
    os.makedirs(target_root)

imgs = os.listdir(source_root)
for img in imgs:
    if img.endswith('.png'):
        print(img)
        names = img.split('_')
        l = len(names)
        target_name = names[0]
        
        for i in range(1,l-1):
            target_name += '_'
            target_name += names[i]
        
        target_name += '.png'
        shutil.copyfile(os.path.join(source_root,img),os.path.join(target_root,target_name))
        
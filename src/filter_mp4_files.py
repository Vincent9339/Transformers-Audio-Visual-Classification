import os, csv
import numpy as np

csv_filename = 'vggsound_filtered.csv'
folder_path = '~/vggsound_08/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video/'




def filter_music_files(csv_filename, folder_path):
    with open(csv_filename, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for i in csv_reader:
            #print(i[0] +'_'+ "%06d" % (int(i[1]),))
            csv_target.append(i[0] +'_'+ "%06d" % (int(i[1]),))
            
    for filename in os.listdir(folder_path):
        #print(filename.split('.')[0])
        mp4_target.append(filename.split('.')[0])
            
    return 

csv_target = list()
mp4_target = list()
filter_music_files(csv_filename, folder_path)
common = list(np.intersect1d(np.array(csv_target), np.array(mp4_target)))
#print(f"all music files: {len(csv_target)}")
#print(f"current len folder: {len(mp4_target)}")
#print(f"len common: {len(common)}")
#print(common[:10])
    
#music_contents = folder_path+'/music_contents/'
#if not os.path.exists(music_contents):
#    os.makedirs(music_contents)

    
for num,filename in enumerate(os.listdir(folder_path)):
    #print(filename.split('.')[0])
    if filename.split('.')[0] in common:
        print(f"{num}; {filename.split('.')[0]}")
        #shutil.copyfile(folder_path+filename, music_contents+filename)
    else: 
        #print(folder_path+filename)
        os.remove(folder_path+filename)
        
        

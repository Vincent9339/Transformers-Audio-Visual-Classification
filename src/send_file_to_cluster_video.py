import csv, re, os, torchaudio, torch

pattern = re.compile('(\s*)playing(\s*)') # this is for removing word playing from vggsound classes
vgg_csv_path='/home/vin/Documents/ITU/thesis/data/vgg/raw/filtered_vggsound.csv'
error_num=0
cnt=0

with open(vgg_csv_path, 'r', newline='') as csv_file:
    csv_reader = csv.reader(csv_file)
    for id_,val,label,_ in csv_reader:
        cnt+=1
        if cnt > 41040:
            l = pattern.sub('',label)
            l = l.replace(',','_')
            l = l.replace(' ','_')
            vid_id = id_+'_'+ "%06d" % (int(val))
            final_dest = '/mnt/d/VggSound/preprocessed/'+l+'/video/'+vid_id+'/'
            print(f"{cnt}, {final_dest}")

            try:
                if vid_id+'.pkl' in os.listdir(final_dest):    
                    os.system("sshpass -p '' scp -rqo LogLevel=QUIET " +final_dest+vid_id+".pkl" + " viaj@hpc.itu.dk:/home/viaj/project/data/vgg/video")
                    
            except:
                error_num+=1

print(error_num)


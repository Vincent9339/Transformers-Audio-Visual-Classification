import csv, re

pattern = re.compile('(\s*)playing(\s*)') # this is for removing word playing from vggsound classes
path='/home/vin/Documents/ITU/thesis/data'
vgg_csv_path=path+'/vgg/raw/'+'filtered_vggsound.csv'

with open(vgg_csv_path, 'r', newline='') as csv_file:
    csv_reader = csv.reader(csv_file)
    for id_,_,label,_ in csv_reader:
        if os.path.exists(path+'/'+'vgg/preprocessed/'+pattern.sub('',label)+'/video/') == False:
            os.makedirs(path+'/'+'vgg/preprocessed/'+pattern.sub('',label)+'/video/')
            os.makedirs(path+'/'+'vgg/preprocessed/'+pattern.sub('',label)+'/audio/')
        if os.path.exists(path+'/'+'vgg/preprocessed/'+pattern.sub('',label)+'/video/'+id_+'_'+ "%06d" % (int(val))) == False:
            os.makedirs(path+'/'+'vgg/preprocessed/'+pattern.sub('',label)+'/video/'+id_+'_'+ "%06d" % (int(val)))
            
            
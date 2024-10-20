import urllib3,shutil, itertools, re, csv
import pandas as pd
import numpy as np

##############################################################################
#               Download vggsound class labels                               #
##############################################################################

url = 'https://huggingface.co/datasets/Loie/VGGSound/raw/main/vggsound.csv'
filename = '/mnt/g/My Drive/Thesis/src/vggsound.csv'
c = urllib3.PoolManager()

with c.request('GET',url, preload_content=False) as resp, open(filename, 'wb') as out_file:
    shutil.copyfileobj(resp, out_file)

##############################################################################
#               Download audioset class_labels_indices                       #
##############################################################################

url = 'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv'
filename = '/mnt/g/My Drive/Thesis/src/class_labels_indices.csv'
c = urllib3.PoolManager()

with c.request('GET',url, preload_content=False) as resp, open(filename, 'wb') as out_file:
    shutil.copyfileobj(resp, out_file)


##############################################################################
#               Filter VGGSound class labels                                 #
##############################################################################

def filter_csv(input_csv_filename, output_csv_filename):
    filtered_rows = list()
    all_names = list()
    not_included = ['volleyball', 'hockey', 'darts', 'tennis', 'badminton', 'table', 'squash']
    with open(input_csv_filename, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        
        for row in csv_reader:
            if row[2].strip().startswith("playing"):
                if row[2].split()[1] in not_included:
                    pass
                else:
                    for i in range(len(row[2].split()[1:])):
                        all_names.append(row[2].split()[1:][i])
                    filtered_rows.append(row)
    #print(f"all musical instrument: {sorted(list(set(all_names)))}")
    with open(output_csv_filename, 'w', newline='') as csv_output_file:
        csv_writer = csv.writer(csv_output_file)
        csv_writer.writerows(filtered_rows)

input_csv_filename = 'vggsound.csv' 
output_csv_filename = 'vggsound_filtered.csv'

filter_csv(input_csv_filename, output_csv_filename)


    
    
##############################################################################
#               Fitler audioset class_labels_indices                         #
##############################################################################
pattern = re.compile('(\s*)playing(\s*)') # this is for removing word playing from vggsound classes

class_file_audioset = 'class_labels_indices.csv'
class_file_vggsound = 'vggsound_filtered.csv' 

audioset_df=pd.read_csv(class_file_audioset,index_col='index')
as_labels=audioset_df['display_name'].tolist()[2:]
as_labels = list(map(lambda x: x.lower(),as_labels))

vggsound_df=pd.read_csv(class_file_vggsound,names=['id', 'U','display_name','type'])
vgg_labels = list(set(vggsound_df['display_name'].tolist()))
vgg_labels=[pattern.sub('',vgg_labels[i]) for i in range(len(vgg_labels))]

common_labels=list(np.intersect1d(np.array(as_labels), np.array(vgg_labels)))

df = audioset_df['display_name'].str.lower()
a=list(set([np.where(df == common_labels[i])[0][0] for i in range(len(common_labels))]))
audioset_df.loc[df.index[a]].to_csv('audioset_filtered.csv')

print("all musical instruments:")
for i in range(len(vgg_labels)):
    print(vgg_labels[i])
    
    
##############################################################################
#               Download extra files for audioset                            #
##############################################################################   
urls = {
    'evaluation' : 'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv',
    'balanced_train' : 'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv',
    'unbalanced_train' : 'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv'
    } 


filename = {
    'evaluation' : 'eval_segments.csv',
    'balanced_train' : 'balanced_train_segments.csv',
    'unbalanced_train' : 'unbalanced_train_segments.csv'

    }

dir_to_save_files = '/mnt/g/My Drive/Thesis/src/'


    c = urllib3.PoolManager()
    with c.request('GET',list(urls.values())[i], preload_content=False) as resp, open(dir_to_save_files+list(filename.values())[i], 'wb') as out_file:
        shutil.copyfileobj(resp, out_file)

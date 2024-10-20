import csv, re, os, torchaudio, torch


def make_features(wav_name, mel_bins=128, target_length=1024):
    waveform, sr = torchaudio.load(wav_name)
    assert sr == 16000, 'input audio sampling rate must be 16kHz'

    
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
        window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=9.8)
    
    n_frames = fbank.shape[0]

    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
    return fbank
    

pattern = re.compile('(\s*)playing(\s*)') # this is for removing word playing from vggsound classes
vgg_csv_path='/home/vin/Documents/ITU/thesis/data/vgg/raw/filtered_vggsound.csv'
error_num=0
cnt=0

with open(vgg_csv_path, 'r', newline='') as csv_file:
    csv_reader = csv.reader(csv_file)
    for id_,val,label,_ in csv_reader:
        cnt+=1
        if cnt > 34159:
            l = pattern.sub('',label)
            l = l.replace(',','_')
            l = l.replace(' ','_')
            aud_id = id_+'_'+ "%06d" % (int(val))
            final_dest = '/mnt/d/VggSound/preprocessed/'+l+'/audio/'
            print(f"{cnt}, {final_dest}")

            try:
                aud = make_features(final_dest+aud_id+'.wav')
                torch.save(aud, '/home/vin/Documents/'+aud_id+'.pt')
                os.system("sshpass -p '' scp -rqo LogLevel=QUIET /home/vin/Documents/"+aud_id+'.pt' + " viaj@hpc.itu.dk:/home/viaj/project/data/vgg/audio")
                os.system("rm /home/vin/Documents/"+aud_id+".pt")
            except:
                error_num+=1

print(error_num)


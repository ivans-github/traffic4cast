import h5py
import tarfile
from fastai.basics import *

############################################################################################

def untar(filename, outpath='../storage/'):
    tarred_file = tarfile.open(filename)
    tarred_file.extractall(path=outpath)
    tarred_file.close()
    return Path('{}{}/'.format(outpath,
                               os.path.basename(filename).split('.')[0]))

############################################################################################

class TrafficDataset(torch.utils.data.Dataset):
    
    def __init__(self, data_folder, city='MOSCOW', partition='training', num_in_frames=12, num_out_frames=12, time_step=1, bs=None):
        
        self.path = Path(f'{data_folder}/{city}/{partition}')
        self.num_in_frames = num_in_frames
        self.num_out_frames = num_out_frames
        self.time_step = time_step
        self.bs = bs
        self.num_frames_perday = 1 + (288 - self.num_in_frames - self.num_out_frames)
        input_mask_loc = Path(f'{data_folder}/{city}/{city}_Mask_5.pt')
        if input_mask_loc.is_file():
            self.mask=torch.load(input_mask_loc)
            print('Using input mask')
        else:
            self.mask=None
            print(f'No input mask found in {input_mask_loc}. Carrying on wihtout applying any input mask')
        
        if not self.path.exists():
            print(f'No input data found in:  {self.path}')
            #untar(f'{data_folder}/{city}.tar')
        
        self.num_days = len(self.path.ls())
        self.filelist = sorted(self.path.ls())
        
        # load static features
        with h5py.File(f'{data_folder}/{city}/{city}_static_2019.h5', 'r') as static_file:
            static_features = static_file.get('array')[()].astype(np.float32)
            static_features = torch.from_numpy(static_features).permute(2, 0, 1)
        self.static_features = static_features
        
        # is_empty and n_inp are required for fastai to work
        self.is_empty = False if len(self)>0 else True
        self.n_inp = 1
    
    def __len__(self):
        if self.bs == None:
            return math.ceil((self.num_days * self.num_frames_perday) / self.time_step)
        else:
            return math.ceil((self.num_days * self.num_frames_perday) / self.time_step) // self.bs
    
    def __getitem__(self, batch_indices):
        
        if not isinstance(batch_indices, Iterable): batch_indices = [batch_indices]
        
        if (self.bs != None) and (len(batch_indices)==1): 
            batch_indices = range(batch_indices[0]*self.bs, (batch_indices[0]+1)*self.bs)
        
        # Add time_step to batch_indices
        batch_indices = [index * self.time_step for index in batch_indices]
        
        indices = collections.defaultdict(list)
        for index in batch_indices:
            indices[index // self.num_frames_perday].append(index % self.num_frames_perday)
        
        batch = []
        for file_idx, frame_idxs in indices.items():
            
            raw = h5py.File(self.filelist[file_idx], 'r').get('array')
            extract = raw[frame_idxs[0]:(frame_idxs[-1] + self.num_in_frames + self.num_out_frames), ...].astype(np.float32)
            
            if self.mask != None:
                extract =np.moveaxis(np.moveaxis(extract, -1,0)*self.mask.numpy(), 0,-1)

            for frame_idx in frame_idxs:
                i = frame_idx - frame_idxs[0]
                
                x = torch.from_numpy(extract[i:i + self.num_in_frames,...]).permute(3, 0, 1, 2)
                C, D, H, W = x.shape
                x = x.reshape(C*D, H, W)
                t = torch.ones(1, H, W) * frame_idx * 255. / (288. - self.num_in_frames) # add time feature (t) - scaled to 255
                x = torch.cat([x, self.static_features, t], dim=0)
                
                y = torch.from_numpy(extract[i + self.num_in_frames:i + self.num_in_frames + self.num_out_frames,
                                             ...]).permute(3, 0, 1, 2)
                y = y[:8, ...]
                
                batch.append((x, y))

        if len(batch) == 1:
            return torch.from_numpy(batch[0][0]), torch.from_numpy(batch[0][1])
        elif self.bs != None:
            return torch.stack([sample[0] for sample in batch]), torch.stack([sample[1] for sample in batch])
        else:
            return batch

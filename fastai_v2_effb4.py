import sys
path_to_fastai_dev_local = '/home/heye0507/fastai_dev'
sys.path.append(path_to_fastai_dev_local)
sys.path.append('/home/jupyter/.local/bin')

from fastai2.basics import *
from fastai2.vision.all import *
from fastai2.medical.imaging import *
from fastai2.callback.tracker import *
from efficientnet_pytorch import EfficientNet

from PIL import Image

np.set_printoptions(linewidth=120)
matplotlib.rcParams['image.cmap'] = 'bone'

path = Path('/home/jupyter/rsns')
path_df = path/'data'
path_trn = path/'data/raw/stage_1_train_images'
path_tst = path/'data/raw/stage_1_test_images'

df_lbls = pd.read_feather(path_df/'labels.fth')
df_tst = pd.read_feather(path_df/'df_tst.fth')
df_trn = pd.read_feather(path_df/'df_trn.fth').dropna(subset=['img_pct_window'])
df_trn['fname'] = df_trn['fname'].apply(lambda x: str(path)+'/data/raw/stage_1_train_images/' + str(Path(x).stem) + '.dcm')
df_tst['fname'] = df_tst['fname'].apply(lambda x: str(path)+'/data/raw/stage_1_test_images/' + str(Path(x).stem) + '.dcm')
comb = df_trn.join(df_lbls.set_index('ID'),'SOPInstanceUID')
df_comb = comb.set_index('SOPInstanceUID')
bins = (path_df/'bins.pkl').load()

bs,nw = 64, 8
set_seed(42)

patients = df_comb.PatientID.unique()
pat_mask = np.random.random(len(patients))<0.8
pat_trn = patients[pat_mask]

def split_data(df):
    idx = L.range(df)
    mask = df.PatientID.isin(pat_trn)
    return idx[mask],idx[~mask]

splits = split_data(df_comb)


def filename(o): return os.path.splitext(os.path.basename(o))[0]

fns = L(list(df_comb.fname)).map(filename)


def fix_pxrepr(dcm):
    if dcm.PixelRepresentation != 0 or dcm.RescaleIntercept<-100: return
    x = dcm.pixel_array + 1000
    px_mode = 4096
    x[x>=px_mode] = x[x>=px_mode] - px_mode
    dcm.PixelData = x.tobytes()
    dcm.RescaleIntercept = -1000
    
def dcm_tfm(fn): 
    fn = (path_trn/fn).with_suffix('.dcm')
    try:
        x = fn.dcmread()
        fix_pxrepr(x)
    except Exception as e:
        print(fn,e)
        raise SkipItemException
    if x.Rows != 512 or x.Columns != 512: x.zoom_to((512,512))
    px = x.scaled_px
    return TensorImage(px.to_3chan(dicom_windows.brain,dicom_windows.subdural, bins=bins))


htypes = ['any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']
def fn2label(fn): return df_comb.loc[fn][htypes].values.astype(np.float32)

def get_data(bs, sz):
    return dsrc.databunch(bs=bs, num_workers=nw, after_batch=batch_tfms+[AffineCoordTfm(size=sz)])

def accuracy_any(inp, targ, thresh=0.5, sigmoid=True):
    inp,targ = flatten_check(inp[:,0],targ[:,0])
    if sigmoid: inp = inp.sigmoid()
    return ((inp>thresh)==targ.bool()).float().mean()


loss_weights = to_device(tensor(2.0, 1, 1, 1, 1, 1))
loss_func = BaseLoss(nn.BCEWithLogitsLoss, pos_weight=loss_weights, floatify=True, flatten=False, 
    is_2d=False, activation=torch.sigmoid)
opt_func = partial(Adam, wd=0.01, eps=1e-3)
metrics=[accuracy_multi,accuracy_any]


aug = aug_transforms(flip_vert=True,max_rotate=60.,p_lighting=0.,max_warp=0.)
tfms = [[dcm_tfm], [fn2label,EncodedMultiCategorize(htypes)]]
dsrc = DataSource(fns, tfms, splits=splits)
nrm = Normalize(tensor([0.485, 0.456, 0.406]), tensor([0.229, 0.224, 0.225]))
batch_tfms = [nrm, Cuda(), *aug]

dbch = get_data(64,256)
model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=6)

learn = Learner(dbch,model,metrics=metrics,
                loss_func=loss_func,
                opt_func=opt_func,
                model_dir=path/'models/eff_net').to_fp16()

lr = 1e-2/2

learn.fit_one_cycle(10,lr,cbs=[EarlyStoppingCallback(min_delta=0.001,patience=3),
                                    SaveModelCallback(every_epoch=True,fname='effb4-v2-256')
                                   ])

learn.save('effb4_v2_256_base')


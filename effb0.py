from fastai.vision import *
from fastai import *
from fastai.data_block import _maybe_squeeze
from efficientnet_pytorch import EfficientNet
from fastai.callbacks import SaveModelCallback,EarlyStoppingCallback

path = Path('/home/jupyter/rsns/data')
df_train = pd.read_csv(path/'train_fastai_format.csv')

bs = 64
sz = 224

# Radek's monkey patch to work out NAN in pandas
def modified_label_from_df(self, cols:IntsOrStrs=1, label_cls:Callable=None, **kwargs):
    "Label `self.items` from the values in `cols` in `self.inner_df`."
    self.inner_df.labels.fillna('', inplace=True)
    labels = self.inner_df.iloc[:,df_names_to_idx(cols, self.inner_df)]
    assert labels.isna().sum().sum() == 0, f"You have NaN values in column(s) {cols} of your dataframe, please fix it."
    if is_listy(cols) and len(cols) > 1 and (label_cls is None or label_cls == MultiCategoryList):
        new_kwargs,label_cls = dict(one_hot=True, classes= cols),MultiCategoryList
        kwargs = {**new_kwargs, **kwargs}
    return self._label_from_list(_maybe_squeeze(labels), label_cls=label_cls, **kwargs)


ItemList.label_from_df = modified_label_from_df
df_test = pd.read_csv(path/'raw/stage_1_sample_submission.csv')
df_test['fn'] = df_test['ID'].apply(lambda x: '_'.join(x.split('_')[0:2]) + '.png')
test_fns = df_test['fn'].unique()

tfms = get_transforms(flip_vert=False,  max_warp=0., max_rotate=60., max_zoom=1.1)

data = (ImageList
        .from_csv(path,'train_fastai_format.csv',folder='preprocessed/224/train')
        .split_by_rand_pct(seed=42)
        .label_from_df(label_delim=' ')
        .transform(tfms,size=(sz,sz))
        .add_test(str(path) + '/preprocessed/224/test/' + test_fns)
        .databunch(bs=bs,num_workers=8)
        .normalize(imagenet_stats)
       )

model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=6)

learn = Learner(data,model,metrics=[accuracy_thresh],model_dir=path/'models/eff_net').to_fp16()

learn.unfreeze()
learn.load('effb0-baseline-128');

lr = 1e-3
learn.fit_one_cycle(10,lr,callbacks=[EarlyStoppingCallback(learn,min_delta=0.001,patience=3),
                                    SaveModelCallback(learn,every='epoch',name='effb0-224')
                                   ])



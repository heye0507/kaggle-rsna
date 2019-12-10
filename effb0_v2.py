from fastai.vision import *
from fastai import *
from fastai.data_block import _maybe_squeeze
from efficientnet_pytorch import EfficientNet

path = Path('/home/jupyter/rsns/data/preprocessed')

#export
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

tfms = get_transforms(flip_vert=True,  max_warp=0., max_rotate=60., max_zoom=1.1,p_lighting=0.)
bs = 64
sz = 128

data = (ImageList
        .from_csv(path,'train_seed42.csv',folder='full_train_jpg')
        .split_from_df(col='is_valid')
        .label_from_df(label_delim=' ')
        .transform(tfms,size=(sz,sz))
        #.add_test(str(path) + '/preprocessed/224/test/' + test_fns)
        .databunch(bs=bs,num_workers=8)
        .normalize(imagenet_stats)
       )

model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=6)
loss_weights = torch.FloatTensor([2,1,1,1,1,1]).cuda()
loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=loss_weights)

learn = Learner(data,model,metrics=[accuracy_thresh],loss_func=loss_func, model_dir=path/'models/eff_net').to_fp16()

from fastai.callbacks import SaveModelCallback#,EarlyStoppingCallback

lr = 1e-3
learn.fit_one_cycle(5,lr,callbacks=[SaveModelCallback(learn,every='epoch',name='effb0-v2')
                                   ])
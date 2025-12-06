_base_ = [
    '../_base_/models/rsos_r101_macvi.py', '../_base_/datasets/lars.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)

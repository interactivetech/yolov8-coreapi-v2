import determined as det
from ultralytics import YOLO
from ultralytics.yolo.v8.detect import DetectionTrainer

dname_to_path = {
     'x-ray-rheumatology':'/run/determined/workdir/shared_fs/andrew-demo-revamp/x-ray-rheumatology/data.yaml',
     'flir-camera-objects': '/run/determined/workdir/shared_fs/andrew-demo-revamp/flir-camera-objects/data.yaml'
    }
    
if __name__ == '__main__':
    info = det.get_cluster_info()
    print("Info")
    print(info)
    print("info.trial.trial_id: ",info.trial.trial_id)
    hparams = info.trial.hparams
    
    print("hparams: ",hparams)
    assert info is not None, "this example only runs on-cluster"
    latest_checkpoint = info.latest_checkpoint
    if latest_checkpoint is None:
        print("latest_checkpoint None: ",latest_checkpoint)
    else:
        print("latest_checkpoint: ",latest_checkpoint)
    
    """Train and optimize YOLO model given training data and device."""
    model ='{}.pt'.format(hparams['model_name'])
    data = dname_to_path[hparams['dataset_name']]  

    args = dict(model=model,
                data=data, 
                batch=hparams['global_batch_size'],
                epochs=hparams['epochs'],
                workers=hparams['workers'],
                imgsz=hparams['imgsz'])
    trainer = DetectionTrainer(overrides=args)
    trainer.train()
import glob
from det_obj_via_images import DetectObjectsInImages

model_path = 'Z:/personal/projects/git/projectIncKnowHow/tf_dev/computerVision/objectDetection/playing_cards' \
                    '/exported-models/ssd/saved_model'
label_path = 'Z:/personal/projects/git/projectIncKnowHow/tf_dev/computerVision/objectDetection/playing_cards' \
                 '/annotations/label_map.pbtxt'
img_path = 'Z:/personal/projects/git/projectIncKnowHow/Data/playing_cards/archive/validation/testing/'
img_path = glob.glob(img_path + '*.*')

objdet = DetectObjectsInImages(model_path, label_path, img_path)

objdet.detect_objects()

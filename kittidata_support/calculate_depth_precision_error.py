import os
from mytool import tool
import numpy as np
from tqdm import tqdm

t = tool()
predict_address = "C:/Users/kk/Desktop/step_330368/"
label_address = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/label_2/"

objects_count = 0
far_objects_count = 0
near_objects_count = 0

hit_objects_count = 0
hit_far_objects_count = 0
hit_near_objects_count = 0

predict_objects_count = 0
predict_far_objects_count = 0
predict_near_objects_count = 0

# poe = predict_objects_error
poe_accumulate = np.zeros((1,7))
poe_far_accumulate = np.zeros((1,7))
poe_near_accumulate = np.zeros((1,7))

predict_idxs = os.listdir(predict_address)

for predict_idx in tqdm(predict_idxs):
    predict_objects = t.get_label(predict_address + predict_idx)
    predict_objects_flag = predict_objects[:,7] >= 0.3
    
    predict_objects = predict_objects[predict_objects_flag][:,0:7]
    label_objects = t.get_label(label_address + predict_idx)[:,0:7]
    
    if len(label_objects) == 0: continue
    
    # pldps = predict_label_distance_pairs
    # row = predict, column = label
    pldps = np.zeros((predict_objects.shape[0], label_objects.shape[0]))
    
    for i in range(predict_objects.shape[0]):
        for j in range(label_objects.shape[0]):
            predict_object = predict_objects[i, 0:3]
            label_object = label_objects[j, 0:3]
            
            pldp = np.abs(predict_object - label_object)
            pldp = np.sqrt(np.sum(pldp * pldp))
            
            pldps[i,j] = pldp.copy()
            
    if predict_objects.shape[0] > label_objects.shape[0]:
        min_distance = pldps.min(1)
        
        for i in range(predict_objects.shape[0] - label_objects.shape[0]):
            predict_objects = np.delete(predict_objects, np.argmax(min_distance), 0)
            pldps = np.delete(pldps, np.argmax(min_distance), 0)
            min_distance = np.delete(min_distance, np.argmax(min_distance), 0)
    
    # plips = predict_label_idx_pairs
    plips = np.argmin(pldps, axis=1)
    
    # poe = predict_objects_error
    poe = predict_objects.copy()
    for i in range(predict_objects.shape[0]):
        poe[i, :] = np.abs(poe[i, :] - label_objects[plips[i], :])
    
    # error value
    poe_accumulate = poe_accumulate + np.sum(poe.copy(), axis=0)
    poe_far_accumulate = poe_far_accumulate + np.sum(poe[predict_objects[:, 2] >= 40.0].copy(), axis=0)
    poe_near_accumulate = poe_near_accumulate + np.sum(poe[predict_objects[:, 2] < 40.0].copy(), axis=0)
    
    # if miss predict
    if len(plips) < label_objects.shape[0]:
        for i in range(label_objects.shape[0]):
            if i in plips:
                continue
            poe_accumulate = poe_accumulate + label_objects[i, :]  
            
            if label_objects[i, 2] >= 40.0:
                poe_far_accumulate = poe_far_accumulate + label_objects[i, :]
            else:
                poe_near_accumulate = poe_near_accumulate + label_objects[i, :]
    
    objects_count = objects_count + label_objects.shape[0]
    far_objects_count = far_objects_count + label_objects[label_objects[:, 2] >= 40.0].shape[0]
    near_objects_count = near_objects_count + label_objects[label_objects[:, 2] < 40.0].shape[0]
    
    # precision (iou threshold = 0.7, equal error length <= 3/17 * box lenght)
    error_threshold = predict_objects[:, 5].copy() * 3 / 17
    
    hit_object = poe[:, 2] <= error_threshold
    hit_objects_count = hit_objects_count + np.sum(hit_object*1)  
    predict_objects_count = predict_objects_count + poe.shape[0]
    
    hit_far_object = hit_object & (predict_objects[:, 2] >= 40.0)
    hit_far_objects_count = hit_far_objects_count + np.sum(hit_far_object*1)
    predict_far_objects_count = predict_far_objects_count + poe[predict_objects[:, 2] >= 40.0].shape[0]
    
    hit_near_object = hit_object & (predict_objects[:, 2] < 40.0)
    hit_near_objects_count = hit_near_objects_count + np.sum(hit_near_object*1)
    predict_near_objects_count = predict_near_objects_count + poe[predict_objects[:, 2] < 40.0].shape[0]
        
mean_error = poe_accumulate[0, 2].copy() / objects_count
mean_precision = hit_objects_count / objects_count

mean_far_error = poe_far_accumulate[0, 2].copy() / far_objects_count
mean_near_error = poe_near_accumulate[0, 2].copy() / near_objects_count

mean_far_precision = hit_far_objects_count / far_objects_count
mean_near_precision = hit_near_objects_count / near_objects_count

predict_percent = predict_objects_count / objects_count
far_predict_percent = predict_far_objects_count / far_objects_count
near_predict_percent = predict_near_objects_count / near_objects_count

print("\n")
print("mean_error: %.4f" % mean_error, "m")
print("mean_precision: %.4f" % mean_precision, "%")
print("\n")
print("mean_far_error: %.4f" % mean_far_error, "m")
print("mean_near_error: %.4f" % mean_near_error, "m")
print("mean_far_precision: %.4f" % mean_far_precision, "%")
print("mean_near_precision: %.4f" % mean_near_precision, "%")
print("\n")
print("predict_percent: %.4f" % predict_percent, "%")
print("far_predict_percent: %.4f" % far_predict_percent, "%")
print("near_predict_percent: %.4f" % near_predict_percent, "%")
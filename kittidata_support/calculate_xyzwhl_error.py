import os
from mytool import tool
import numpy as np
from tqdm import tqdm

t = tool()
predict_address = "C:/Users/kk/Desktop/rebuttal/dfrank/"
label_address = "C:/Users/kk/Desktop/rebuttal/label/"

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
        
    poe_flag = poe > 0.1
    poe_flag = poe_flag*1
    poe_flag2 = poe < 4 # 4
    poe_flag2 = poe_flag2*1
    poe = poe*poe_flag2
    
    # error value
    poe_accumulate = poe_accumulate + np.sum(poe.copy(), axis=0)
    poe_far_accumulate = poe_far_accumulate + np.sum(poe[predict_objects[:, 2] >= 40.0].copy(), axis=0)
    poe_near_accumulate = poe_near_accumulate + np.sum(poe[predict_objects[:, 2] < 40.0].copy(), axis=0)

    # # if miss predict
    # if len(plips) < label_objects.shape[0]:
    #     for i in range(label_objects.shape[0]):
    #         if i in plips:
    #             continue
    #         poe_accumulate = poe_accumulate + np.abs(label_objects[i, :]  )
            
    #         if label_objects[i, 2] >= 40.0:
    #             poe_far_accumulate = poe_far_accumulate + label_objects[i, :]
    #         else:
    #             poe_near_accumulate = poe_near_accumulate + label_objects[i, :]
    
    objects_count = objects_count + label_objects.shape[0]
    far_objects_count = far_objects_count + label_objects[label_objects[:, 2] >= 40.0].shape[0]
    near_objects_count = near_objects_count + label_objects[label_objects[:, 2] < 40.0].shape[0]
    
        
mean_x_error = poe_accumulate[0, 0].copy() / objects_count
mean_x_far_error = poe_far_accumulate[0, 0].copy() / far_objects_count
mean_x_near_error = poe_near_accumulate[0, 0].copy() / near_objects_count

mean_y_error = poe_accumulate[0, 1].copy() / objects_count
mean_y_far_error = poe_far_accumulate[0, 1].copy() / far_objects_count
mean_y_near_error = poe_near_accumulate[0, 1].copy() / near_objects_count

mean_z_error = poe_accumulate[0, 2].copy() / objects_count
mean_z_far_error = poe_far_accumulate[0, 2].copy() / far_objects_count
mean_z_near_error = poe_near_accumulate[0, 2].copy() / near_objects_count

mean_w_error = poe_accumulate[0, 3].copy() / objects_count
mean_w_far_error = poe_far_accumulate[0, 3].copy() / far_objects_count
mean_w_near_error = poe_near_accumulate[0, 3].copy() / near_objects_count

mean_h_error = poe_accumulate[0, 4].copy() / objects_count
mean_h_far_error = poe_far_accumulate[0, 4].copy() / far_objects_count
mean_h_near_error = poe_near_accumulate[0, 4].copy() / near_objects_count

mean_l_error = poe_accumulate[0, 5].copy() / objects_count
mean_l_far_error = poe_far_accumulate[0, 5].copy() / far_objects_count
mean_l_near_error = poe_near_accumulate[0, 5].copy() / near_objects_count

print("\n")
print("mean_x_error: %.4f" % poe_accumulate[0, 0], "m")
print("mean_y_error: %.4f" % poe_accumulate[0, 1], "m")
print("mean_z_error: %.4f" % poe_accumulate[0, 2], "m")
print("mean_w_error: %.4f" % poe_accumulate[0, 3], "m")
print("mean_h_error: %.4f" % poe_accumulate[0, 4], "m")
print("mean_l_error: %.4f" % poe_accumulate[0, 5], "m")
# print("mean_far_error: %.4f" % mean_far_error, "m")
# print("mean_near_error: %.4f" % mean_near_error, "m")

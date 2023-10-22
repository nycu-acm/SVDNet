import numpy as np
import os

count = True

# diag = np.sqrt(1.6**2 + 3.9**2)

for idx in os.listdir("C:/Users/kk/Desktop/AutonomousCar/PointPillar/combine_ratio_free/"):
    print(idx)
    predict_combine_ratio = np.load("C:/Users/kk/Desktop/AutonomousCar/PointPillar/combine_ratio_free/"  + idx)
    
    if count:
        predict_combine_ratios = predict_combine_ratio[np.newaxis,:,:].copy()
        count = False
        continue
    
    predict_combine_ratios = np.concatenate((predict_combine_ratios,predict_combine_ratio[np.newaxis,:,:]), axis=0)

# predict_combine_ratios[:,0] = predict_combine_ratios[:,0] * diag
# predict_combine_ratios[:,1] = predict_combine_ratios[:,1] * diag
# predict_combine_ratios[:,2] = predict_combine_ratios[:,2] * 1.56
# predict_combine_ratios[:,3] = predict_combine_ratios[:,3] * diag
# predict_combine_ratios[:,4] = predict_combine_ratios[:,4] * diag
# predict_combine_ratios[:,5] = predict_combine_ratios[:,5] * 1.56

# distance1_residual = predict_combine_ratios[predict_combine_ratios[:,6]<=100][:, 0:6]
# distance2_residual = predict_combine_ratios[(predict_combine_ratios[:,6]>100) & (predict_combine_ratios[:,6]<=200)][:, 0:6]
# distance3_residual = predict_combine_ratios[(predict_combine_ratios[:,6]>200) & (predict_combine_ratios[:,6]<=300)][:, 0:6]
# distance4_residual = predict_combine_ratios[predict_combine_ratios[:,6]>300][:, 0:6]

distance1_residual_mean = np.mean(predict_combine_ratios, axis=0)
# distance2_residual_mean = np.mean(distance2_residual, axis=0)
# distance3_residual_mean = np.mean(distance3_residual, axis=0)
# distance4_residual_mean = np.mean(distance4_residual, axis=0)

distance1_residual_var = np.var(predict_combine_ratios, axis=0)
# distance2_residual_var = np.var(distance2_residual, axis=0)
# distance3_residual_var = np.var(distance3_residual, axis=0)
# distance4_residual_var = np.var(distance4_residual, axis=0)
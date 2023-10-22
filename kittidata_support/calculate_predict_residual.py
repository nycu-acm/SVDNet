import numpy as np
import os

count = True

diag = np.sqrt(1.6**2 + 3.9**2)

for idx in os.listdir("C:/Users/kk/Desktop/pred_residual/"):
    print(idx)
    predict_residual = np.load("C:/Users/kk/Desktop/pred_residual/"  + idx)
    predict_residual = np.delete(predict_residual, [3,4,5,6,10,11,12,13], axis=1)
    
    if count:
        predict_residuals = predict_residual.copy()
        count = False
        continue
    
    predict_residuals = np.concatenate((predict_residuals,predict_residual), axis=0)

predict_residuals[:,0] = predict_residuals[:,0] * diag
predict_residuals[:,1] = predict_residuals[:,1] * diag
predict_residuals[:,2] = predict_residuals[:,2] * 1.56
predict_residuals[:,3] = predict_residuals[:,3] * diag
predict_residuals[:,4] = predict_residuals[:,4] * diag
predict_residuals[:,5] = predict_residuals[:,5] * 1.56

distance1_residual = predict_residuals[predict_residuals[:,6]<=100][:, 0:6]
distance2_residual = predict_residuals[(predict_residuals[:,6]>100) & (predict_residuals[:,6]<=200)][:, 0:6]
distance3_residual = predict_residuals[(predict_residuals[:,6]>200) & (predict_residuals[:,6]<=300)][:, 0:6]
distance4_residual = predict_residuals[predict_residuals[:,6]>300][:, 0:6]

distance1_residual_mean = np.mean(distance1_residual, axis=0)
distance2_residual_mean = np.mean(distance2_residual, axis=0)
distance3_residual_mean = np.mean(distance3_residual, axis=0)
distance4_residual_mean = np.mean(distance4_residual, axis=0)

distance1_residual_var = np.var(distance1_residual, axis=0)
distance2_residual_var = np.var(distance2_residual, axis=0)
distance3_residual_var = np.var(distance3_residual, axis=0)
distance4_residual_var = np.var(distance4_residual, axis=0)
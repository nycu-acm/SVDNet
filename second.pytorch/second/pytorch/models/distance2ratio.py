import torch
import torch.nn as nn
import torch.nn.functional as F

# # output_result_two2D_learned_ratiosum_function_pretrain
# class distance2ratio(nn.Module):
#     def __init__(self, out_features=5):
#         super(distance2ratio, self).__init__()
        
#         self.pridict1 = torch.nn.Conv2d(64, 16, (2,2), (2,2))
        
#         self.pridict2 = torch.nn.Conv2d(16, 1, (1,1), 1)
        
#         self.fc1 = torch.nn.Linear(in_features=216, out_features=out_features, bias=False)
        

#     def forward(self, x):
#         x = self.pridict1(x)
#         x = self.pridict2(x)
#         x = torch.mean(x, 2)
#         x = x.view(-1, 216)
#         x = self.fc1(x)
#         x = F.softmax(x)
#         return x
    
    
# output_result_three2D_learned_ratiosum_function_pretrain
    
class distance2ratio(nn.Module):
    def __init__(self, out_features=5):
        super(distance2ratio, self).__init__()
        
        self.pridict1 = torch.nn.Conv2d(64, 16, (2,2), (2,2))
        
        self.pridict2 = torch.nn.Conv2d(16, 1, (1,1), 1)
        
        self.fc1 = torch.nn.Linear(in_features=216, out_features=out_features, bias=False)
        self.fc2 = torch.nn.Linear(in_features=216, out_features=1, bias=False)
        

    def forward(self, x):
        x = self.pridict1(x)
        x = self.pridict2(x)
        x = torch.mean(x, 2)
        x = x.view(-1, 216)
        y = self.fc2(x)
        y = F.sigmoid(y)
        x = self.fc1(x)
        x = F.softmax(x)
        return x, y
    
class only_distance2ratio(nn.Module):
    def __init__(self,):
        super(only_distance2ratio, self).__init__()
        
        self.pridict = torch.nn.Conv2d(29, 1, (1,1), (1,1))
        

    def forward(self, x):
        x = self.pridict(x)
        x = torch.mean(x, 2)
        x = F.sigmoid(x)*0.5
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F

class Tnet(nn.Module):
    def __init__(self, k=3):
        super(Tnet, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
    
    def forward(self, x):
        B, D, N = x.size()
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out = torch.amax(out, 2)
        
        out = F.relu(self.bn4(self.fc1(out)))
        out = F.relu(self.bn5(self.fc2(out)))
        out = self.fc3(out)
        
        iden = torch.eye(self.k, device=x.device).view(1, self.k*self.k).repeat(B, 1)
        out = out + iden
        out = out.view(-1, self.k, self.k)
        
        return out

class Transform(nn.Module):
    def __init__(self, k_input=3, k_feature=64, global_feature=True, feature_transform=True):
        super(Transform, self).__init__()
        self.global_feature = global_feature
        self.feature_transform = feature_transform
        
        self.input_transform = Tnet(k=k_input)
        self.feature_transform = Tnet(k=k_feature)
        
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
    def forward(self, x):
        B, D, N = x.size()
        
        input_transform = self.input_transform(x)
        x = torch.bmm(input_transform, x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        
        if self.feature_transform:
            feature_transform = self.feature_transform(out)
            out = torch.bmm(feature_transform, out)
        else:
            feature_transform = None
        
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out = torch.amax(out, 2)
        # return out
        if not self.global_feature:
            out = out.unsqueeze(-1).repeat(1, 1, N)
        return out, input_transform, feature_transform

def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss

# ------ TO DO ------
class cls_model(nn.Module):
    def __init__(self, k=2):
        super(cls_model, self).__init__()
        self.k = k
        self.feature_transform = Transform()
        
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        
    def forward(self, points, debug=True):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        B, N, K = points.size()
        points = points.permute(0, 2, 1)
        
        transform = self.feature_transform(points)
        

        # out_max = F.max_pool1d(out_conv4, N).squeeze(dim=-1)
        out_max = torch.amax(out_conv4, dim=-1) #[0]
        out = self.net(out_max)
        if debug:
            print("points.shape: ", points.shape)
            print("out_conv1.shape: ", out_conv1.shape)
            print("out_conv2.shape: ", out_conv2.shape)
            print("out_conv3.shape: ", out_conv3.shape)
            print("out_conv4.shape: ", out_conv4.shape)
            print("out_max.shape: ", out_max.shape)
            print("out.shape: ", out.shape)
            print()
        return out
    
# ------ TO DO ------
class seg_model(nn.Module):
    def __init__(self, num_seg_classes = 6):
        super(seg_model, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.conv4 = nn.Conv1d(128, 1024, 1)
        
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.batch_norm4 = nn.BatchNorm1d(1024)
        
        self.net = nn.Sequential(
            nn.Conv1d(1088, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            # nn.Dropout(p=0.3),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # nn.Dropout(p=0.3),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.Dropout(p=0.3),
            nn.Conv1d(128, num_seg_classes, 1)
        )

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''
        B, N, _ = points.size()
        out_conv1 = F.relu(self.batch_norm1(self.conv1(points.permute(0, 2, 1))))
        out_conv2 = F.relu(self.batch_norm2(self.conv2(out_conv1)))
        out_conv3 = F.relu(self.batch_norm3(self.conv3(out_conv2)))
        out_conv4 = F.relu(self.batch_norm4(self.conv4(out_conv3)))
        
        # out_max = F.max_pool1d(out_conv4, N).squeeze(dim=-1)
        out_max = torch.amax(out_conv4, dim=-1)
        out_max = out_max.unsqueeze(-1).repeat(1, 1, N)
        out = self.net(torch.cat([out_conv4, out_max], dim=1))
        
        # global_out = torch.amax(global_out, dim=-1, keepdims=True).repeat(1, 1, N)
        # out = torch.cat((local_out, global_out), dim=1)
        # out = self.point_layer(out).transpose(1, 2)
        
        return out




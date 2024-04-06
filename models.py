import torch
import torch.nn as nn
import torch.nn.functional as F

# ------ TO DO ------
class cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_model, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.conv4 = nn.Conv1d(128, 1024, 1)
        
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.batch_norm4 = nn.BatchNorm1d(1024)
                
        self.net = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        B, N, _ = points.size()
        out_conv1 = F.relu(self.batch_norm1(self.conv1(points.permute(0, 2, 1))))
        out_conv2 = F.relu(self.batch_norm2(self.conv2(out_conv1)))
        out_conv3 = F.relu(self.batch_norm3(self.conv3(out_conv2)))
        out_conv4 = F.relu(self.batch_norm4(self.conv4(out_conv3)))

        # out_max = F.max_pool1d(out_conv4, N).squeeze(dim=-1)
        out_max = torch.amax(out_conv4, dim=-1) #[0]
        out = self.net(out_max)
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




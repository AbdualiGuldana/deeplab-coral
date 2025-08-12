# import torch
# from torch import nn
# from config import SENSOR_RATIO
# from ..sensor import BaseSensor


# class SensorVisionFusion(nn.Module):
#     def __init__(self, channels=2048):
#         super(SensorVisionFusion, self).__init__()
        
#         # Initialize the sensor model
#         self.s_model = BaseSensor(channels)
        
#         # Define a new learnable fusion layer
#         # It takes the concatenated features (visual + sensor) and combines them.
#         # The input channels will be 'channels' (from visual) + 'channels' (from sensor).
#         self.fusion_conv = nn.Sequential(
#             nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, v_features, sensors):
#         # Process the sensor data to get a feature map
#         s_features = self.s_model(sensors)
        
#         # Ensure the sensor feature map has the same spatial dimensions as the visual features
#         # This is important for concatenation.
#         s_features_upsampled = F.interpolate(
#             s_features, 
#             size=v_features.shape[-2:], 
#             mode="bilinear", 
#             align_corners=False
#         )
        
#         # Concatenate the visual and upsampled sensor features along the channel dimension
#         concatenated_features = torch.cat((v_features, s_features_upsampled), dim=1)
        
#         # Pass the concatenated features through the learnable fusion layer
#         fused_feature = self.fusion_conv(concatenated_features)
        
#         return fused_feature



from torch import nn

from config import SENSOR_RATIO

from ..sensor import BaseSensor


class SensorVisionFusion(nn.Module):
    def __init__(self, channels=2048):
        super(SensorVisionFusion, self).__init__()
        self.s_model = BaseSensor(channels)

        pass

    def forward(self, v_features, sensors):
        s_features = self.s_model(sensors)

        fused_feature = v_features + (s_features * SENSOR_RATIO)

        return fused_feature

# 模型结构的代码
import torch
from torch import nn
import torch.nn.functional as F
from toolbox.MRI.complex_conver import complex2real,real2complex
from toolbox.MRI.fft import fft2, ifft2

class UNet(nn.Module):
    def contracting_block(self, in_channels, out_channels, kernel_size=3):
        padding = (kernel_size - 1) // 2
        kernel_size = (kernel_size, kernel_size, kernel_size)
        block = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels,
                            padding=padding),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels,
                            padding=padding),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.LeakyReLU(),
        )
        return block

    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        padding = (kernel_size - 1) // 2
        kernel_size = (kernel_size, kernel_size, kernel_size)
        block = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel,
                            padding=padding),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.LeakyReLU(),
            # torch.nn.BatchNorm3d(mid_channel),
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel,
                            padding=padding),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.LeakyReLU(),
            # torch.nn.BatchNorm3d(mid_channel),
            torch.nn.ConvTranspose3d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2,
                                     padding=1, output_padding=1)  # 上采样
        )
        return block

    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        padding = (kernel_size - 1) // 2
        kernel_size = (kernel_size, kernel_size, kernel_size)
        block = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel,
                            padding=padding),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.LeakyReLU(),
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel,
                            padding=padding),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.LeakyReLU(),
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
            # torch.nn.BatchNorm3d(out_channels),
            # torch.nn.LeakyReLU(),
        )
        return block

    def __init__(self, in_channel, out_channel, channels=None, res=True):
        super(UNet, self).__init__()
        self.res = res
        if channels is None:
            channels = [16, 32, 64, 128, 64, 32, 16]
        elif len(channels) != 9:
            raise ValueError('The length of channels of UNet will be assume as 8')
        # Encode
        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=channels[0])
        self.conv_maxpool1 = torch.nn.MaxPool3d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(channels[0], channels[1])
        self.conv_maxpool2 = torch.nn.MaxPool3d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(channels[1], channels[2])
        self.conv_maxpool3 = torch.nn.MaxPool3d(kernel_size=2)
        # self.conv_encode4 = self.contracting_block(channels[2], channels[3])
        # self.conv_maxpool4 = torch.nn.MaxPool3d(kernel_size=2)
        # Bottleneck
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=(3, 3, 3), in_channels=channels[2], out_channels=channels[3],
                            padding=(3 - 1) // 2),
            torch.nn.BatchNorm3d(channels[3]),
            torch.nn.LeakyReLU(),
            torch.nn.Conv3d(kernel_size=(3, 3, 3), in_channels=channels[3], out_channels=channels[3],
                            padding=(3 - 1) // 2), 
            torch.nn.BatchNorm3d(channels[3]),
            torch.nn.LeakyReLU(),
            # torch.nn.BatchNorm3d(512),
            torch.nn.ConvTranspose3d(in_channels=channels[3], out_channels=channels[4], kernel_size=(3, 3, 3), stride=2,
                                     padding=1,
                                     output_padding=1)
        )
        # Decode
        # self.conv_decode3 = self.expansive_block(channels[3], channels[4], channels[5])
        self.conv_decode2 = self.expansive_block(channels[3], channels[4], channels[5])
        self.conv_decode1 = self.expansive_block(channels[4], channels[5], channels[6])
        self.final_layer = self.final_block(channels[5], channels[6], out_channel)

    def crop_and_concat(self, upsampled, bypass, crop=False):
        if crop:
            # c = (bypass.size()[2] - upsampled.size()[2]) / 2
            slice_c = (bypass.size()[2] - upsampled.size()[2]) / 2
            # 这里为了适配数据，暂时设置成 左1 上1 前1
            if slice_c != 0:
                upsampled = F.pad(upsampled, (0, 0, 0, 0, 1, 0))
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x):
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)
        # Bottleneck
        bottleneck1 = self.bottleneck(encode_pool3)
        # Decode
        decode_block3 = self.crop_and_concat(bottleneck1, encode_block3, crop=True)
        cat_layer2 = self.conv_decode2(decode_block3)
        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2, crop=True)
        cat_layer1 = self.conv_decode1(decode_block2)
        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1, crop=True)
        final_layer = self.final_layer(decode_block1)
        return final_layer


class Model(nn.Module):
    def __init__(self,
                 n_input_channels=2,
                 n_output_channels=2,
                 channels=None,
                 res=False
                 ):
        super().__init__()
        self.n_input_channels = n_input_channels
        self.n_output_channels = n_output_channels
        self.res = res
        self.unet = UNet(self.n_input_channels, self.n_output_channels, channels, self.res)

    def forward(self, inputs):
        """
        :param inputs: [batch_size real/imag frame nx  ny]
        :return: unet_complex(inputs) [batch_size real/imag frame nx  ny]
        """
        inputs = inputs[0] # B F X Y 
        complex_type = inputs.type()
        inputs = complex2real(inputs,dim=1)
        output = self.unet(inputs)  # [batch_size real/imag frame nx  ny]
        output = real2complex(output,dim=1,complex_type=complex_type)
        output = torch.abs(output)
        return [output]

class Model_with_DC(nn.Module): #DC
    def __init__(self,
                 n_input_channels=2,
                 n_output_channels=2,
                 channels=None,
                 res=False,
                 init_lambda = 1e6
                 ):
        
        super().__init__()
        self.n_input_channels = n_input_channels
        self.n_output_channels = n_output_channels
        self.res = res
        self.unet = UNet(self.n_input_channels, self.n_output_channels, channels, self.res)
        self.dc_lambda = torch.nn.Parameter(torch.tensor(init_lambda),requires_grad=True)

    def forward(self, inputs):
        """
        :param inputs: [combined_undersampling_image,csm,undersampling_image],[batch_size real/imag frame nx  ny]
        :return: unet_complex(inputs) [batch_size real/imag frame nx  ny]
        """
        combined_undersampling_image,csm,undersampling_kdata,sampling_mask = inputs
        inv_sampling_mask = (sampling_mask == 0).to(torch.int).to(sampling_mask.device)
        # inputs = inputs[0] # B F X Y 
        complex_type = combined_undersampling_image.type()
        combined_undersampling_image = complex2real(combined_undersampling_image,dim=1)
        output = self.unet(combined_undersampling_image)  # [batch_size real/imag frame nx  ny]
        output = real2complex(output,dim=1,complex_type=complex_type) # B F X Y 
        """
        DC
        """
        output = output.unsqueeze(2) * csm # B F C X Y 
        output_kdata = fft2(output) # B F C X Y 
        output_kdata = output_kdata*inv_sampling_mask + (output_kdata*sampling_mask + self.dc_lambda * (undersampling_kdata*sampling_mask)) / (1+self.dc_lambda)
        output = ifft2(output_kdata)
        output = torch.sum(output * csm.conj(),dim=2)
        output = torch.abs(output)
        return [output]
    

if __name__ == '__main__':
    device = torch.device('cpu')  # if torch.cuda.is_available() else torch.device('cpu')
    model = Model(2, 2, res=False).to(device)
    tensor_ones = torch.ones([4, 2, 12, 256, 256], device=device) # batch_size real/imag frame nx  ny
    result = model(tensor_ones)
    print(result.shape)
    print(None)

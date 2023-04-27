import torch.nn as nn
import numpy as np
import torch


class EMGNet(nn.Module):

    def __init__(self, args):
        super(EMGNet, self).__init__()
        self.num_channels = args.num_channels
        self.num_forces = args.num_forces
        self.num_force_levels = args.num_force_levels
        self.num_frames = args.num_frames
        self.num_frequencies = args.num_frequencies
        self.window_length = args.window_length
        self.hop_length = args.hop_length
        # STFT
        hann_window = torch.hann_window(self.window_length)
        if args.cuda:
            hann_window = hann_window.cuda()
        self.hann_window = hann_window
        # Layers
        self.encoder = self._encoder()
        self.decoder = self._decoder()
        self.linear = nn.Linear(32, self.num_force_levels * self.num_forces)

    def _encoder(self):
        layers = []
        layers.append(self._encoder_block(self.num_channels, 32, (2, 4)))
        layers.append(self._encoder_block(32, 128, (2, 4)))
        layers.append(self._encoder_block(128, 256, (2, 4)))
        return nn.Sequential(*layers)

    def _encoder_block(self, in_channel, out_channel, downsample):
        layers = []
        layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=1, padding='same', bias=False))
        layers.append(nn.BatchNorm2d(out_channel))
        layers.append(nn.ReLU(inplace=False))
        layers.append(nn.MaxPool2d(kernel_size=downsample, stride=downsample))
        return nn.Sequential(*layers)

    def _decoder(self):
        layers = []
        layers.append(self._decoder_block(256, 256, (self.num_frames // 4, 1)))
        layers.append(self._decoder_block(256, 128, (self.num_frames // 2, 1)))
        layers.append(self._decoder_block(128, 32, (self.num_frames, 1)))
        return nn.Sequential(*layers)

    def _decoder_block(self, in_channel, out_channel, upsample):
        layers = []
        layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=1, padding='same', bias=False))
        layers.append(nn.BatchNorm2d(out_channel))
        layers.append(nn.ReLU(inplace=False))
        layers.append(nn.Upsample(size=upsample, mode='bilinear', align_corners=False))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size * self.num_channels, x.size(2))
        x = torch.stft(x, n_fft=self.window_length, hop_length=self.hop_length, win_length=self.window_length, window=self.hann_window,
                       center=False, pad_mode='reflect', normalized=False, onesided=True, return_complex=True).abs()
        x = x.view(batch_size, self.num_channels, x.size(1), x.size(2))
        x = x.transpose(2, 3)[..., :self.num_frequencies]
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(x.size(0), x.size(1), x.size(2))
        x = x.transpose(1, 2)
        x = self.linear(x)
        x = x.transpose(1, 2)
        x = x.view(x.size(0), self.num_force_levels, self.num_forces, x.size(2))
        return x


if __name__ == "__main__":
    pass

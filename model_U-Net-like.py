import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    # Двойной сверточный блок => [Conv2d -> BN -> ReLU] * 2
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            # padding=1 сохраняет размер при kernel_size=3
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), 
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    # Downscaling с MaxPool затем ConvBlock
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), 
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    # Upscaling затем ConvBlock
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ConvBlock(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = ConvBlock(in_channels, out_channels) 

    def forward(self, x1, x2):
        # x1 - тензор с слоя декодера (более глубокий)
        # x2 - тензор со слоя кодера (skip connection)
        x1 = self.up(x1)
        # Вход: [N, C_in, H, W] , Цель: [N, C_in, H*2, W*2]

        # Обработка несовпадения размеров из-за пулинга/апсемплинга
        # Если H/W у x1 не совпадает с H/W у x2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # Паддинг x1 до размера x2
        # (отступ слева, отступ справа, отступ сверху, отступ снизу)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # Конкатенация по канальному измерению
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    # Финальная свертка 1x1 для получения нужного числа каналов
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels_in=1, n_channels_out=1, bilinear=True):
        super(UNet, self).__init__()
        if not (isinstance(n_channels_in, int) and n_channels_in > 0):
             raise ValueError("n_channels_in должно быть положительным целым числом")
        if not (isinstance(n_channels_out, int) and n_channels_out > 0):
             raise ValueError("n_channels_out должно быть положительным целым числом")

        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.bilinear = bilinear

        # Кодер
        self.inc = ConvBlock(n_channels_in, 64) # Начальный блок
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        # Определяем factor для ConvTranspose2d если не используется bilinear
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor) # Бутылочное горлышко

        # Декодер 
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_channels_out) # Выходной слой

    def forward(self, x):
        # Кодер
        x1 = self.inc(x)     # -> [N, 64, H, W]
        x2 = self.down1(x1)  # -> [N, 128, H/2, W/2]
        x3 = self.down2(x2)  # -> [N, 256, H/4, W/4]
        x4 = self.down3(x3)  # -> [N, 512, H/8, W/8]
        x5 = self.down4(x4)  # -> [N, 512(или 1024), H/16, W/16] (bottleneck)

        # Декодер
        x = self.up1(x5, x4) # Конкатенация x4 -> [N, 512 или 1024] -> Up -> [N, 256 или 512, H/8, W/8]
        x = self.up2(x, x3)  # Конкатенация x3 -> [N, 256 или 512] -> Up -> [N, 128 или 256, H/4, W/4]
        x = self.up3(x, x2)  # Конкатенация x2 -> [N, 128 или 256] -> Up -> [N, 64 или 128, H/2, W/2]
        x = self.up4(x, x1)  # Конкатенация x1 -> [N, 64 или 128]  -> Up -> [N, 64, H, W]
        logits = self.outc(x)# -> [N, n_channels_out, H, W]
        
        return logits

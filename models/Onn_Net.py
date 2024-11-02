
import os
import torch.nn
import torch
from torch import pi
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Diffraction(torch.nn.Module):
    def __init__(self, lam, size):
        super(Diffraction, self).__init__()
        # optical basic variable
        # light
        self.lam = lam
        self.k = 2 * pi / self.lam
        # diffraction fringe
        self.pixel_size = 0.875/1e6

        # model
        self.size = size.clone().detach()

        # k-space coordinate
        self.u = torch.fft.fftshift(torch.fft.fftfreq(self.size[0], self.pixel_size))
        self.v = torch.fft.fftshift(torch.fft.fftfreq(self.size[1], self.pixel_size))
        self.fu, self.fv = torch.meshgrid(self.u, self.v, indexing='xy')

        # frequency response function
        self.h = lambda fu, fv, wl, z: \
            torch.exp((1.0j * 2 * pi * z / wl) * torch.sqrt(1 - (wl * fu) ** 2 - (wl * fv) ** 2))


        self.limit = 1 / self.lam

    def light_forward(self, images, distance):
        k_images = torch.fft.fft2(images)
        k_images = torch.fft.fftshift(k_images, dim=(2, 3))

        mask_input = torch.hypot(self.fu, self.fv) < self.limit
        h_input_limit = (mask_input * self.h(self.fu, self.fv, self.lam, distance )).cuda()
        k_output = k_images * h_input_limit

        output = torch.fft.ifft2(k_output)
        return output

    def lens_forward(self, images, f):
        proportional = torch.exp(
            ((1.0j * self.k / (2 * f)) * ((self.lam * self.fu) ** 2 + (self.lam * self.fv) ** 2))
        ) / (1.0j * f * self.lam)
        k_images = torch.fft.fft2(images)
        output = proportional * torch.fft.fftshift(k_images, dim=(2, 3))
        base_phase = self.k * torch.tensor(f)
        random_diffuser = -base_phase + 2 * pi * torch.rand(self.size[0], self.size[1], device=device)

        return output, random_diffuser

    # ##################################################################################################################


class Diffuser(torch.nn.Module):
    def __init__(self, n0, n1, lam, size):
        super(Diffuser, self).__init__()
        self.n0 = n0
        self.n1 = n1
        self.delta_n = self.n1 - self.n0
        self.lam = lam
        self.size = torch.tensor(size)


    def diffuserplane(self, miu, sigma0, sigma):

        kernel = self.gaussian_2d_kernel(
            torch.tensor((torch.sqrt(2 * torch.log(torch.tensor(2))) * sigma) / self.lam ,
                         dtype=torch.int16) * 2 + 1, sigma).unsqueeze(0).unsqueeze(0)

        diffuser = torch.FloatTensor(torch.normal(miu, sigma0,
                                                  size=(self.size[0] + kernel.shape[2] - 1,
                                                        self.size[1] + kernel.shape[3] - 1))).unsqueeze(0).unsqueeze(0)

        return torch.conv2d(diffuser, kernel, padding=0).squeeze(0).squeeze(0)

    def gaussian_2d_kernel(self, kernel_size, Sigma):
        kernel = torch.zeros([kernel_size, kernel_size])
        center = kernel_size // torch.tensor(2)

        x = torch.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
        y = torch.linspace(kernel_size // 2, -kernel_size // 2, kernel_size)
        mask_x, mask_y = torch.meshgrid(x, y, indexing='xy')
        mask_rho = torch.hypot(mask_x, mask_y)
        mask = (mask_rho < center)

        if Sigma == 0:
            sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8

        s = 2 * (Sigma ** 2)
        for i in range(0, kernel_size):
            for j in range(0, kernel_size):
                x = (i - center) * self.lam
                y = (j - center) * self.lam

                kernel[i, j] = torch.exp(torch.div(-(x ** 2 + y ** 2), s))

        kernel = kernel * mask
        kernel = kernel / torch.sum(kernel)

        return torch.FloatTensor(kernel)


class Onn_Net(torch.nn.Module):
    def __init__(self, num_layers, size, lam):
        super(Onn_Net, self).__init__()
        self.num_layers = num_layers
        self.size = torch.tensor(size)
        self.lam = lam

        self.phase_modulator = []
        for layer in range(self.num_layers):
            self.phase_modulator.append(
                torch.nn.Parameter(torch.rand(size=(self.size[0], self.size[1]))))
            self.register_parameter(f'phase_modulator_{layer}', self.phase_modulator[layer])

    def forward(self, inputs):

        diffraction = Diffraction(lam=self.lam, size=self.size)
        x = diffraction.light_forward(inputs, 50/1e6)


        # ##############################################################################################################

        for index, phase in enumerate(self.phase_modulator[:-1]):
            x = x * torch.exp(1.0j * 2 * pi * torch.sigmoid(phase))
            x = diffraction.light_forward(x, 50/1e6)
        x = x * torch.exp(1.0j * 2 * pi * torch.sigmoid(self.phase_modulator[-1]))


        # ##############################################################################################################

        '''输出层'''
        x = diffraction.light_forward(x, 50/1e6)
        x = torch.abs(x) ** 2


        return x


if __name__ == '__main__':
    x1 = torch.rand([4, 1, 240, 240])
    label1 = torch.rand([4, 1, 240, 240])
    models = Onn_Net(num_layers=4, size=(240, 240), lam=5e-9)

    # for xy in models.state_dict():
    #     print(xy, models.state_dict()[xy])
    # print('*' * 50)
    # for xy in models.parameters():
    #     print(xy)
    # print('*' * 50)

    loss_function = torch.nn.MSELoss()

    prediction1 = models(x1)
    m1 = torch.mean(prediction1)
    s1 = torch.std(prediction1)
    loss = loss_function(prediction1, label1)
    print(loss)

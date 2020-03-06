#! /usr/bin/env python3

# Standard Python
import os
import argparse
import logging

# Thirdparty
import torch
from tqdm import tqdm
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms.functional import normalize, to_pil_image
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

# Local
import settings
from config import Config
from modules import StyleTransfer

logging.basicConfig(level=logging.INFO, format='\n%(message)s\n')
logging.getLogger().setLevel(logging.INFO)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info(f'Starting training with device {device}')


class Training:
    def __init__(self):
        self.cfg = cfg
        self.st = StyleTransfer(cfg)
        self.sw = SummaryWriter()
        self._set_initial_image()
        self.step = 0

    def _set_initial_image(self):
        if cfg.random_initial_image:
            self.x = torch.rand(1,
                                3,
                                self.cfg.image_size,
                                self.cfg.image_size,
                                device=self.cfg.device,
                                requires_grad=True)
        else:
            self.x = self.st.content_image.clone().requires_grad_(True).to(
                self.cfg.device)

    def closure(self):
        self.step += 1
        _, M, N = self.st.content_image_features.shape
        with torch.no_grad():
            self.x.data.clamp_(0., 1.).requires_grad_(True)
        self.st.forward(self.x)
        content_loss = F.mse_loss(self.st.content_image_features,
                                  self.st.content_features)
        style_loss = torch.zeros(1).to(self.cfg.device)
        tot_style_layers = len(self.st.style_features)
        for idx in range(tot_style_layers - self.cfg.num_style_layers,
                         tot_style_layers):
            layer_features = self.st.style_features[idx].squeeze(0).to(
                self.cfg.device)
            M_s, N_s = layer_features.shape
            gram_x = torch.mm(layer_features, layer_features.t()).div(
                M_s * N_s).to(self.cfg.device)
            style_loss = style_loss.add(
                torch.mul(
                    1.0 / self.cfg.num_style_layers,
                    F.mse_loss(
                        self.st.style_image_gram_matrices[idx],
                        gram_x,
                    )))

        combined_loss = torch.add(
            torch.mul(self.st.params['alpha'], content_loss),
            torch.mul(self.st.params['beta'], style_loss))

        combined_loss.backward()
        with torch.no_grad():
            self.x.data.clamp_(0., 1.)

        logs = {
            'content_loss': content_loss,
            'style_loss': style_loss,
            'combined_loss': combined_loss
        }
        for k, v in logs.items():
            self.sw.add_scalar(k, v, self.step)

        if self.step % 50 == 0:
            self.sw.add_image(f'result', self.x.squeeze(0), self.step)
        return combined_loss

    def apply_style_transfer(self, cfg):
        opt = torch.optim.LBFGS([self.x], lr=cfg.learning_rate)
        for step in tqdm(range(self.cfg.steps)):
            opt.step(self.closure)
            opt.zero_grad()
        self.sw.add_image(f'result', self.x.squeeze(0), self.step)

        def _is_random_start():
            return '_random' if self.cfg.random_initial_image else ''

        result_fpath = os.path.join(
            settings.IMG_DIR,
            f'result_{os.path.splitext(os.path.split(self.cfg.content_image_path)[-1])[0]}\
_{os.path.splitext(os.path.split(self.cfg.style_image_path)[-1])[0]}{_is_random_start()}.jpg'
        )
        if self.cfg.normalize:
            self.x = normalize(
                self.x.squeeze(0),
                self.cfg.denorm_mean,
                self.cfg.denorm_std,
            )
            self.x = self.x.unsqueeze(0)
        save_image(self.x, result_fpath)
        logging.info(f'Done! New image saved at {result_fpath}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "content_image_fname",
        type=str,
        default=None,
        help=
        "Input content image filename. Image must be placed in the /img directory"
    )
    parser.add_argument(
        "style_image_fname",
        type=str,
        default=None,
        help=
        "Input style image filename. Image must be placed in the /img directory"
    )
    parser.add_argument("-g",
                        "--steps",
                        type=int,
                        default=150,
                        help="Max number of optimization steps. Default: 150")
    parser.add_argument("-s",
                        "--image_size",
                        type=int,
                        default=256,
                        help="Size of output image. Default: 256x256")
    parser.add_argument(
        "-l",
        "--num_style_layers",
        type=int,
        default=5,
        help="Number of convolutional layers to use as style layers. Default: 5"
    )
    parser.add_argument(
        "-r",
        "--random_initial_image",
        action='store_true',
        help="Initialize output image at random. Default: False")
    parser.add_argument(
        "-a",
        "--alpha",
        type=int,
        default=1,
        help=
        "Parameter alpha, controlling the weight of the content loss in the total loss. Default: 1"
    )
    parser.add_argument(
        "-b",
        "--beta",
        type=int,
        default=5e5,
        help=
        "Parameter beta, controlling the weight of the style loss in the total loss. Default: 5e5"
    )
    parser.add_argument(
        "-n",
        "--normalize",
        action='store_true',
        help=
        "Normalize input with ImageNet mean and standard deviation. Default: False"
    )

    args = parser.parse_args()
    content_image_path, style_image_path = [
        os.path.join(settings.IMG_DIR,
                     os.path.split(f)[-1])
        for f in (getattr(args, a)
                  for a in ['content_image_fname', 'style_image_fname'])
    ]

    for img in [content_image_path, style_image_path]:
        if not os.path.exists(img):
            raise Exception(
                f'Image {img} not found. Please enter a valid image path')

    cfg = Config(
        device,
        content_image_path,
        style_image_path,
        args.steps,
        args.image_size,
        {
            'alpha': args.alpha,
            'beta': args.beta,
        },
        args.random_initial_image,
        args.num_style_layers,
        args.normalize,
    )

    t = Training()
    t.apply_style_transfer(cfg)

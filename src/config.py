class Config:
    def __init__(
        self,
        device,
        content_image_path,
        style_image_path,
        steps,
        image_size,
        params,
        random_initial_image,
        num_style_layers,
        normalize,
        num_workers=0,
        learning_rate=0.95,
        content_layer_idx=0,
        norm_mean=[0.485, 0.456, 0.406],
        norm_std=[0.229, 0.224, 0.225],
    ):
        self.device = device
        self.content_image_path = content_image_path
        self.style_image_path = style_image_path
        self.steps = steps
        self.image_size = image_size,
        self.num_workers = num_workers
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.content_layer_idx = content_layer_idx
        self.num_style_layers = num_style_layers
        self.normalize = normalize
        self.params = params
        self.random_initial_image = random_initial_image
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.denorm_mean = [
            -m / self.norm_std[i] for i, m in enumerate(self.norm_mean)
        ]
        self.denorm_std = [1.0 / s for s in self.norm_std]
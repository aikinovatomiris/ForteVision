"""ResNet18 feature extractor. Only explicit local files; never downloads."""
from pathlib import Path

from .dataset import ForteError, sha256
from .preprocessing import SamplingConfig, check_deadline, iter_frames, pooled_windows, probe


class ResNetExtractor:
    architecture = 'resnet18'
    weights_name = 'IMAGENET1K_V1'

    def __init__(self, weights_path: Path, device='cpu', image_mode='center'):
        weights_path = Path(weights_path)
        if not weights_path.is_file():
            raise ForteError('Нет локальных весов ResNet18. Выполните python -m scripts.prepare_weights заранее.')
        import torch
        from torchvision.models import ResNet18_Weights, resnet18
        if device not in ('cpu', 'cuda'):
            raise ForteError('device должен быть cpu или cuda')
        if device == 'cuda' and not torch.cuda.is_available():
            raise ForteError('CUDA запрошена, но недоступна')
        if image_mode not in ('center', 'full_frame'):
            raise ForteError('Unknown image preprocessing mode')
        self.image_mode = image_mode
        self.torch, self.device = torch, torch.device(device)
        torch.set_num_threads(min(4, torch.get_num_threads()))
        self.weights_sha256 = sha256(weights_path)
        if not self.weights_sha256.startswith('f37072fd'):
            raise ForteError('Файл не соответствует torchvision ResNet18 IMAGENET1K_V1 (SHA-256)')
        network = resnet18(weights=None)
        try:
            network.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True), strict=True)
        except Exception as exc:
            raise ForteError(f'Поврежденные или несовместимые веса ResNet18: {exc}') from exc
        network.fc = torch.nn.Identity()
        self.model = network.to(self.device).eval()
        self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms()

    def preprocess(self, frame):
        from PIL import Image, ImageOps
        image = Image.fromarray(frame[:, :, ::-1].copy())
        if self.image_mode == 'full_frame':
            from torchvision.transforms.functional import to_tensor, normalize
            image = ImageOps.pad(image, (320, 320), color=(124, 116, 104))
            return normalize(to_tensor(image), self.transform.mean, self.transform.std)
        return self.transform(image)

    def embed(self, tensors):
        tensor = self.torch.stack(tensors).to(self.device)
        with self.torch.no_grad():
            return self.model(tensor).cpu().numpy()

    def extract(self, path: Path, config: SamplingConfig, deadline=None):
        info = probe(path)
        frames, times, all_times, embeddings = [], [], [], []
        for timestamp, frame in iter_frames(path, config, deadline):
            frames.append(self.preprocess(frame))
            times.append(timestamp)
            if len(frames) == config.batch_size:
                embeddings.extend(self.embed(frames))
                all_times.extend(times)
                frames, times = [], []
                check_deadline(deadline)
        if frames:
            embeddings.extend(self.embed(frames))
            all_times.extend(times)
        check_deadline(deadline)
        x, windows = pooled_windows(all_times, embeddings, config, info['duration_seconds'])
        return x, windows, dict(**info, sampled_frames=len(all_times), timestamp_source='opencv_pos_msec')

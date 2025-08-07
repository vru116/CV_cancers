import torch
import torch.nn.functional as F
import numpy as np

class AbsCAMInit:
    def __init__(self, model, target_layer='layer4', device='cuda'):
        self.model = model
        self.device = device
        self.target_layer = target_layer
        self._register_hooks()

    def _register_hooks(self):
        self.activations = None
        self.gradients = None

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                break
        else:
            raise ValueError(f"Layer {self.target_layer} not found in model")

    def __call__(self, input_tensor, class_idx=None):
        input_tensor = input_tensor.to(self.device)
        self.model.zero_grad()

        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        score = output[:, class_idx]
        score.backward()

        gradients = self.gradients
        # print(gradients.shape)
        activations = self.activations

        weights = gradients.abs().mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)

        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        cam -= cam.min()
        cam /= cam.max() + 1e-8
        # print("S0:", cam.min().item(), cam.max().item(), cam.mean().item())
        return cam




class AbsCAMFinal:
    def __init__(self, model, target_layer='layer4', device='cuda'):
        self.model = model
        self.device = device
        self.target_layer = target_layer
        self._register_hooks()

    def _register_hooks(self):
        self.activations = None
        self.gradients = None

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                break
        else:
            raise ValueError(f"Layer {self.target_layer} not found in model")

    def __call__(self, input_tensor, class_idx=None):
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True
        self.model.zero_grad()

        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        score = output[:, class_idx]
        score.backward(retain_graph=True)

        gradients = self.gradients
        activations = self.activations

        weights = gradients.abs().mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        # cam = F.relu(cam)
        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]), mode='bilinear', align_corners=False)
        cam_norm = cam - cam.min()
        cam_norm = cam_norm / (cam_norm.max() + 1e-8)
        M0 = cam_norm 
        # print("M0:", M0.min().item(), M0.max().item(), M0.mean().item())

        M1 = input_tensor * M0.repeat(1, 3, 1, 1)
        # print("M1:", M1.min().item(), M1.max().item(), M1.mean().item())
        with torch.no_grad():
            output_M1 = self.model(M1)
            # prob = torch.softmax(output_M1, dim=1)
            yc_M1 = output_M1[:, class_idx]
            # print(f"yc_M1 = {yc_M1.item():.4f}")

        # print(f"yc_M1 = {yc_M1.item():.4f}, M0 mean = {M0.mean().item():.4f}")

        Lc = F.relu(yc_M1.item() * M0.squeeze())
        return Lc.cpu().numpy()
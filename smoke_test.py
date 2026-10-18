import torch, torchvision, pandas, sklearn, pydicom, gradio
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
x = torch.randn(2, 3, 224, 224)
m = torchvision.models.resnet18(weights=None)
print("forward ok:", m(x).shape)
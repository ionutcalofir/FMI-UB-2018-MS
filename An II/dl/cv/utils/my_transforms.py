import torchvision.transforms as transforms

my_transforms = {
  'to_pil': transforms.ToPILImage(),
  'to_tensor': transforms.ToTensor(),
  'normalize_imagenet': transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  'resize': transforms.Resize(512),
  'hflip': transforms.RandomHorizontalFlip(),
  'vflip': transforms.RandomVerticalFlip(),
  'center_crop': transforms.CenterCrop(224),
  'random_crop': transforms.RandomCrop(224),
  'random_erase': transforms.RandomErasing()
}

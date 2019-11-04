import cv2 as cv
import numpy as np
import torch
from torchvision import transforms

from data_gen import data_transforms


def main():
    transformer = data_transforms['valid']

    # checkpoint = 'image_classification.pt'
    # print('loading model: {}...'.format(checkpoint))
    # model = ImgClsModel()
    # model.load_state_dict(torch.load(checkpoint))
    # model = model.to(device)
    # model.eval()

    scripted_quantized_model_file = 'mobilenet_quantization_scripted_quantized.pth'
    model = torch.jit.load(scripted_quantized_model_file)
    model = model.to('cpu')
    model.eval()

    full_path = 'images/test_image.jpg'
    img = cv.imread(full_path)

    img = img[..., ::-1]  # RGB
    img = transforms.ToPILImage()(img)
    img = transformer(img)
    img = torch.unsqueeze(img, dim=0)
    img = img.to('cpu')

    with torch.no_grad():
        out = model(img)

    out = out.cpu().numpy()[0]
    out = np.argmax(out, axis=-1).astype(np.int).tolist()
    print(out)


if __name__ == "__main__":
    main()

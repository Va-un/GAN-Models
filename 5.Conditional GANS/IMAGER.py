def IMG_GEN(img_grid_real,img_grid_fake):
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    # Assuming img_grid_real and img_grid_fake are torch tensors
    # Convert the tensors to NumPy arrays and reshape if needed
    img_grid_real_np = img_grid_real.detach().cpu().numpy()
    img_grid_fake_np = img_grid_fake.detach().cpu().numpy()

    # Reshape if necessary (remove batch dimension if present)
    img_grid_real_np = img_grid_real_np[0] if len(img_grid_real_np.shape) == 4 else img_grid_real_np
    img_grid_fake_np = img_grid_fake_np[0] if len(img_grid_fake_np.shape) == 4 else img_grid_fake_np

    # Convert from CHW to HWC format (assuming the image is in CHW format)
    img_grid_real_np = np.transpose(img_grid_real_np, (1, 2, 0))
    img_grid_fake_np = np.transpose(img_grid_fake_np, (1, 2, 0))

    # Denormalize the pixel values if normalization was applied during image generation
    # Assuming the images were normalized to the range [0, 1]
    img_grid_real_np = img_grid_real_np * 255
    img_grid_fake_np = img_grid_fake_np * 255

    # Convert NumPy arrays to PIL images
    img_real = Image.fromarray(img_grid_real_np.astype('uint8'))
    img_fake = Image.fromarray(img_grid_fake_np.astype('uint8'))

    # Display the images using Matplotlib
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img_real)
    plt.title("Real Images")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_fake)
    plt.title("Fake Images")
    plt.axis('off')
    plt.savefig("generated_images.png")
    plt.show()
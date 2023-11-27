
from amd.rocal.plugin.generic import ROCALClassificationIterator
from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types 
import cv2
import cupy as cp
import random
import numpy as np
import test as t

def generate_random_numbers(count):
    """Generate a list of random floating-point numbers."""
    random_numbers = []
    for _ in range(count):
        random_numbers.append(random.uniform(1.0, 100.0))  # Generates random floats between 1.0 and 100.0
    return random_numbers

def generate_random_numbers1(count):
    import random
    """Generate a list of random numbers."""
    random_numbers = []
    for _ in range(count):
        random_numbers.append(random.randint(1, 100))  # Generates random integers between 1 and 100 (inclusive)
    return random_numbers

def draw_patches(img, idx, device):
    # image is expected as a tensor, bboxes as numpy
    if device == "gpu":
        img = cp.asnumpy(img)
    # Ensure the image has a compatible depth (e.g., CV_8U) before saving
    img = img.astype(np.uint8)  # Convert to 8-bit unsigned integers
    # img = img.transpose([0, 2, 3, 1])
    images_list = []
    for im in img:
        images_list.append(im)
    img = cv2.vconcat(images_list)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("eso_blur_host" + str(idx) + ".png", img,
                [cv2.IMWRITE_PNG_COMPRESSION, 9])


def main():
    # Create Pipeline instance
    batch_size = 5
    num_threads = 1
    device_id = 0
    local_rank = 0
    world_size = 1
    rocal_cpu = True
    random_seed = 0
    max_height = 720
    max_width = 640
    color_format = types.RGB
    data_path="/media/MIVisionX-data/rocal_data/coco/coco_10_img/train_10images_2017/"
    decoder_device = 'cpu'
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=random_seed, rocal_cpu=rocal_cpu, tensor_layout=types.NHWC , tensor_dtype=types.FLOAT, output_memory_type=types.HOST_MEMORY if rocal_cpu else types.DEVICE_MEMORY)
    with pipe:
        jpegs, _ = fn.readers.file(file_root=data_path)
        images = fn.decoders.image(jpegs,
                                    file_root=data_path,
                                    device=decoder_device,
                                    max_decoded_width=max_width,
                                    max_decoded_height=max_height,
                                    output_type=color_format,
                                    shard_id=local_rank,
                                    num_shards=world_size,
                                    random_shuffle=False)
        output = fn.external_source(images, source = generate_random_numbers, size=batch_size)
        # output1 = fn.external_source(images, source = generate_random_numbers, size=batch_size)
        # output = t.external_source(images, source = generate_random_numbers, size=batch_size)
        # exit(0)
        contrast_output = fn.contrast(images, contrast_center=output, contrast = output)
        # blur_output = fn.blur(images, window_size=output1)

        pipe.set_outputs(contrast_output)
    pipe.build()

    # Dataloader
    data_loader = ROCALClassificationIterator(
        pipe, device="cpu", device_id=local_rank)
    cnt = 0

    # Enumerate over the Dataloader
    for epoch in range(int(1)):
        print("EPOCH:::::", epoch)
        for i, (output_list, labels) in enumerate(data_loader, 0):
            for j in range(len(output_list)):
                print("**************", i, "*******************")
                print("**************starts*******************")
                print("\nImages:\n", output_list[j])
                print("\nLABELS:\n", labels)
                print("**************ends*******************")
                print("**************", i, "*******************")
                draw_patches(output_list[j], cnt, "cpu")
                cnt += len(output_list[j])

        data_loader.reset()

if __name__ == '__main__':
    main()
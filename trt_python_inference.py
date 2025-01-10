import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import os
import cv2

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def preprocess_image(image_path):
    im = cv2.imread(image_path).astype(np.float32)
    assert im is not None, f'Image Not Found {image_path}'
    h0, w0 = im.shape[:2] # 1080, 1920 
    r = 640 / max(h0, w0)  # ratio # 640 / 1920
    interp = cv2.INTER_LINEAR # if (r > 1) else cv2.INTER_AREA
    im = cv2.resize(im, (int(w0 * r), int(h0 * r)), interpolation=interp)
    img, ratio, pad = letterbox(im, [384, 640], auto=False, scaleup=False)
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    # img = torch.from_numpy(img) / 255.0
    img = img / 255.0
    return img[None, :], im

def build_engine_from_onnx(onnx_file_path, engine_file_path):
    """Build a TensorRT engine from an ONNX file and save it."""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    parser = trt.OnnxParser(network, logger)

    # Parse the ONNX file
    with open(onnx_file_path, "rb") as f:
        if not parser.parse(f.read()):
            print("Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # Build and serialize the engine
    print("Building the TensorRT engine...")
    engine = builder.build_engine(network, config)
    if engine is not None:
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
        print(f"Serialized engine saved to {engine_file_path}")
    else:
        print("Failed to build the engine.")
    return engine

def load_engine(engine_path):
    """Load a TensorRT engine."""
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise ValueError(f"Failed to load TensorRT engine from {engine_path}")
        return engine

def allocate_buffers(engine):
    """Allocate host and device buffers for the engine."""
    h_inputs = []
    h_outputs = []
    d_inputs = []
    d_outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the corresponding buffers to device bindings
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            h_inputs.append(host_mem)
            d_inputs.append(device_mem)
        else:
            h_outputs.append(host_mem)
            d_outputs.append(device_mem)

    return h_inputs, d_inputs, h_outputs, d_outputs, bindings, stream

def do_inference(context, bindings, h_inputs, d_inputs, h_outputs, d_outputs, stream, input_data):
    """Perform inference using the TensorRT engine."""
    # Transfer input data to the GPU
    cuda.memcpy_htod_async(d_inputs[0], input_data, stream)
    # Run inference
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU
    for h_output, d_output in zip(h_outputs, d_outputs):
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
    # Synchronize the stream
    stream.synchronize()
    return h_outputs

def main():
    onnx_file_path = "/home/ubuntu/autocam/tycho_yolo_v6.1.1copy/onnxmodel"  # Replace with your ONNX model path
    engine_file_path = "/home/ubuntu/autocam/tycho_yolo_v6.1.1copy/onnxmodel_model.engine"

    # Check if the engine file exists; if not, create it
    if not os.path.exists(engine_file_path):
        print("Engine file not found. Building engine...")
        engine = build_engine_from_onnx(onnx_file_path, engine_file_path)
        if engine is None:
            print("Failed to create the engine.")
            return
    else:
        print("Engine file found. Loading engine...")
        engine = load_engine(engine_file_path)

    # Allocate buffers
    h_inputs, d_inputs, h_outputs, d_outputs, bindings, stream = allocate_buffers(engine)

    # Create execution context
    context = engine.create_execution_context()

    # Prepare input data (example, replace with actual input)
    # input_data = np.random.random_sample(trt.volume(engine.get_binding_shape(0))).astype(np.float32)
    # Prepare input data (load and preprocess image)
    image_path = "test_image.png"  # Replace with your image path
    input_shape = engine.get_binding_shape(0)  # Get input shape from engine

    # Ensure input shape is CHW format and adjust dimensions
    height, width = input_shape[2], input_shape[3]
    channels = input_shape[1]

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image at {image_path}")
        return

    # Resize the image to match the model input
    image = cv2.resize(image, (width, height))

    # Convert to RGB if necessary
    if channels == 3 and image.shape[2] == 3:  # Assuming model expects RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize and transpose image to CHW format
    image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
    input_data = np.transpose(image, (2, 0, 1))  # HWC to CHW
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
    # Make the input data contiguous in memory
    input_data = np.ascontiguousarray(input_data)

    input_data, original_image = preprocess_image(image_path)


    # Perform inference
    outputs = do_inference(context, bindings, h_inputs, d_inputs, h_outputs, d_outputs, stream, input_data)

    # User specifies which output to print
    print("Available outputs:")
    for i, binding in enumerate(engine):
        if not engine.binding_is_input(binding):  # Only output bindings
            print(f"Index {i}: {binding}")

    user_output_index = int(input(f"Enter output index (0 to {len(outputs)-1}): "))
    
    if 0 <= user_output_index < len(outputs):
        selected_output = outputs[user_output_index]
        print(f"Output {user_output_index}:")
        print(selected_output)
        print(f"Sum of Output {user_output_index}: {np.sum(selected_output)}")
        cv2.imwrite("./test_image_output.png", np.reshape(selected_output*255, (48,80)))
    else:
        print("Invalid index. Please try again.")

if __name__ == "__main__":
    main()
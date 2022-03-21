import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

engine_path = '../person.engine'
trt_logger = trt.Logger(trt.Logger.INFO)
runtime = trt.Runtime(trt_logger)
with open(engine_path, 'rb') as f:
    engine = runtime.deserialize_cuda_engine(f.read())

# create context
context = engine.create_execution_context()
INPUT_DATA_TYPE = np.float32
stream = cuda.Stream()

# 在内存（Host）中分配空间
host_in = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=INPUT_DATA_TYPE)
host_out = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=INPUT_DATA_TYPE)

# 在显存（Device）中分配空间
devide_in = cuda.mem_alloc(host_in.nbytes)
devide_out = cuda.mem_alloc(host_out.nbytes)

bindings = [int(devide_in), int(devide_out)]


# 如果输入输出已经确定
img=''
np.copyto(host_in, img.ravel())
cuda.memcpy_htod_async(devide_in, host_in, stream)
context.execute_async(bindings=bindings, stream_handle=stream.handle)
cuda.memcpy_dtoh_async(host_out, devide_out, stream)
stream.synchronize()

# 如果输入输出数量不一定
# 参考 https://github.com/NVIDIA/TensorRT/blob/master/samples/python/common.py
# Transfer input data to the GPU.
[cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
# Run inference.
context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
# Transfer predictions back from the GPU.
[cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
# Synchronize the stream
stream.synchronize()
# Return only the host outputs.
return [out.host for out in outputs]

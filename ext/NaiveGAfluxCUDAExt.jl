module NaiveGAfluxCUDAExt

using NaiveGAflux
import CUDA

struct NaiveGAfluxCudaDevice{D}
    device::D
end

# Doesn't seem like we can do much with the device we got from CUDA, but lets keep it in case someone finds a use for it
NaiveGAflux.execution_device(a::CUDA.CuArray) = NaiveGAfluxCudaDevice(CUDA.device(a))

function NaiveGAflux._availablebytes(::NaiveGAfluxCudaDevice)
    # Doesn't seem like CUDA exposes these per device
    info = CUDA.MemoryInfo()
    info.free_bytes + info.pool_reserved_bytes - info.pool_used_bytes
end

# Should map data to device, but how?
NaiveGAflux.matchdatatype(::NaiveGAfluxCudaDevice, iter) = GpuIterator(iter)

NaiveGAflux._rangetoarr(a::Type{<:CUDA.CuArray}) = a

end
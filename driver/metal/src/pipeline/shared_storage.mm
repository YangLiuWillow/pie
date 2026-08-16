#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "pipeline/shared_storage.hpp"

#include <cstring>

namespace pie::metal::pipeline {

namespace {

/// The system default device, fetched ONCE for the process.
///
/// `MTLCreateSystemDefaultDevice()` is not a getter. Measured on this machine
/// it costs 0.13-0.17 ms per call, against 0.002 ms for the
/// `newBufferWithLength:` it exists to serve -- the lookup is ~50x the
/// allocation it precedes.
///
/// That mattered because this function is called TWICE PER CHANNEL (the cell
/// ring and its four control words) and a decode program registers ten
/// channels on every bind. Twenty lookups at 0.14 ms is 2.7 ms, which was
/// ~99% of `register_channel_set` and, since the guest rebinds its program
/// once per token, about 15% of a decode step -- larger than any single
/// kernel in it. See `docs/NEXT-decode-dispatch-count.md`.
///
/// A function-local static is initialized once under C++11's thread-safe
/// initialization, and the returned device is a process-wide singleton, so
/// holding it for the process lifetime is what every other Metal entry point
/// here already does with its own `id<MTLDevice>`.
id<MTLDevice> shared_device() {
    static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return device;
}

}  // namespace

SharedStorage make_platform_shared_storage(std::size_t size) {
    SharedStorage storage;
    if (size == 0) return storage;

    id<MTLDevice> device = shared_device();
    if (device == nil) return storage;
    id<MTLBuffer> buffer =
        [device newBufferWithLength:size options:MTLResourceStorageModeShared];
    if (buffer == nil) return storage;
    std::memset(buffer.contents, 0, size);

    void* retained = (__bridge_retained void*)buffer;
    storage.owner = std::shared_ptr<void>(
        retained,
        [](void* ptr) {
            if (ptr == nullptr) return;
            id released = (__bridge_transfer id)ptr;
            (void)released;
        });
    storage.contents = static_cast<std::uint8_t*>(buffer.contents);
    storage.native_buffer = (__bridge void*)buffer;
    storage.gpu_address = buffer.gpuAddress;
    storage.size = size;
    return storage;
}

}  // namespace pie::metal::pipeline

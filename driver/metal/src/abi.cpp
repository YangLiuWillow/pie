#include "context.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <string>

#include "pie/driver/validate.hpp"

namespace {

pie::metal::Context* as_context(PieDriver* driver) {
    return reinterpret_cast<pie::metal::Context*>(driver);
}

PieDriver* create_context(const PieDriverCreateDesc& desc, PieDriverCaps* caps) {
    std::memset(caps, 0, sizeof(*caps));
    const std::string config_path(
        reinterpret_cast<const char*>(desc.config_bytes.ptr),
        desc.config_bytes.len);
    auto context = std::make_unique<pie::metal::Context>();
    if (context->initialize(config_path, desc.runtime) != PIE_STATUS_OK) {
        return nullptr;
    }
    context->fill_device_facts(caps);
    return reinterpret_cast<PieDriver*>(context.release());
}

extern "C" int32_t pie_metal_load_model(
    PieDriver* driver,
    const PieModelLoadDesc* load,
    PieDriverCaps* caps) {
    const int status = pie::driver::validate::model_load_desc(load, caps);
    if (status != PIE_STATUS_OK) return status;
    if (driver == nullptr) return PIE_STATUS_INVALID_ARGUMENT;
    try {
        std::memset(caps, 0, sizeof(*caps));
        return as_context(driver)->load_model(*load, caps);
    } catch (const std::exception& e) {
        std::cerr << "[pie-driver-metal] load_model: " << e.what() << "\n";
        return PIE_STATUS_DRIVER_ERROR;
    } catch (...) {
        std::cerr << "[pie-driver-metal] load_model: unknown exception\n";
        return PIE_STATUS_DRIVER_ERROR;
    }
}

}  // namespace

extern "C" PieDriver* pie_metal_create(
    const PieDriverCreateDesc* desc,
    PieDriverCaps* caps) {
    if (pie::driver::validate::create_desc(desc, caps) != PIE_STATUS_OK) {
        return nullptr;
    }
    try {
        return create_context(*desc, caps);
    } catch (const std::exception& e) {
        std::cerr << "[pie-driver-metal] create: " << e.what() << "\n";
        return nullptr;
    } catch (...) {
        std::cerr << "[pie-driver-metal] create: unknown exception\n";
        return nullptr;
    }
}

extern "C" int32_t pie_metal_register_program(
    PieDriver* driver,
    const PieProgramDesc* program,
    std::uint64_t* program_id) {
    const int status = pie::driver::validate::program_desc(program, program_id);
    if (status != PIE_STATUS_OK) return status;
    if (driver == nullptr) return PIE_STATUS_INVALID_ARGUMENT;
    try {
        return as_context(driver)->register_program(*program, program_id);
    } catch (const std::exception& e) {
        std::cerr << "[pie-driver-metal] register_program: " << e.what() << "\n";
        return PIE_STATUS_DRIVER_ERROR;
    } catch (...) {
        std::cerr << "[pie-driver-metal] register_program: unknown exception\n";
        return PIE_STATUS_DRIVER_ERROR;
    }
}

extern "C" int32_t pie_metal_register_channel(
    PieDriver* driver,
    const PieChannelDesc* channel,
    PieChannelEndpointBinding* binding) {
    const int status = pie::driver::validate::channel_desc(channel, binding);
    if (status != PIE_STATUS_OK) return status;
    if (driver == nullptr) return PIE_STATUS_INVALID_ARGUMENT;
    try {
        return as_context(driver)->register_channel(*channel, binding);
    } catch (const std::exception& e) {
        std::cerr << "[pie-driver-metal] register_channel: " << e.what() << "\n";
        return PIE_STATUS_DRIVER_ERROR;
    } catch (...) {
        std::cerr << "[pie-driver-metal] register_channel: unknown exception\n";
        return PIE_STATUS_DRIVER_ERROR;
    }
}

extern "C" int32_t pie_metal_bind_instance(
    PieDriver* driver,
    const PieInstanceDesc* instance,
    PieInstanceBinding* binding) {
    const int status = pie::driver::validate::instance_desc(instance, binding);
    if (status != PIE_STATUS_OK) return status;
    if (driver == nullptr) return PIE_STATUS_INVALID_ARGUMENT;
    try {
        return as_context(driver)->bind_instance(*instance, binding);
    } catch (const std::exception& e) {
        std::cerr << "[pie-driver-metal] bind_instance: " << e.what() << "\n";
        return PIE_STATUS_DRIVER_ERROR;
    } catch (...) {
        std::cerr << "[pie-driver-metal] bind_instance: unknown exception\n";
        return PIE_STATUS_DRIVER_ERROR;
    }
}

/// Why a frame was refused, when `pie_metal_launch` returns
/// `PIE_STATUS_INVALID_ARGUMENT`.
///
/// `validate::frame_desc` answers with a bare status code and no indication of
/// WHICH of its dozen checks failed, so a rejected frame surfaces to the guest
/// as `direct launch rejected: pie_metal_launch failed with status -1` and
/// nothing more. That is enough to know a descriptor is malformed and useless
/// for knowing how -- which cost this investigation two wrong hypotheses
/// (pool exhaustion, fire width) before anyone looked here.
///
/// Set `PIE_METAL_FRAME_DIAG=1` to have a refusal describe the frame it
/// refused. Off by default: this runs on the launch path.
static void diagnose_frame(const PieFrameDesc* f) {
    static const bool on = [] {
        const char* e = std::getenv("PIE_METAL_FRAME_DIAG");
        return e != nullptr && *e != '\0' && !(e[0] == '0' && e[1] == '\0');
    }();
    if (!on || f == nullptr) return;
    std::fprintf(stderr, "[frame-diag] REFUSED abi=%u roster=%zu steps=%zu "
                 "kv_xlat=%zu kv_indptr=%zu reserved=(%u,%u)\n",
                 unsigned(f->abi_version), std::size_t(f->instance_ids.len),
                 std::size_t(f->steps.len), std::size_t(f->kv_translation.len),
                 std::size_t(f->kv_translation_indptr.len),
                 unsigned(f->reserved0), unsigned(f->reserved1));
    // The roster, and whether it repeats -- the one check in `frame_desc` that
    // a MULTI-LANE frame can trip while every single-lane frame passes.
    std::fprintf(stderr, "[frame-diag]   instance_ids = [");
    for (std::size_t i = 0; i < f->instance_ids.len && i < 32; ++i) {
        std::fprintf(stderr, "%llu%s",
                     (unsigned long long)f->instance_ids.ptr[i],
                     i + 1 < f->instance_ids.len ? ", " : "");
    }
    std::fprintf(stderr, "]\n");
    for (std::size_t i = 0; i < f->instance_ids.len; ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            if (f->instance_ids.ptr[i] == f->instance_ids.ptr[j]) {
                std::fprintf(stderr, "[frame-diag]   DUPLICATE instance id %llu "
                             "at slots %zu and %zu -- frame_desc rejects this\n",
                             (unsigned long long)f->instance_ids.ptr[i], j, i);
            }
        }
    }
}

extern "C" int32_t pie_metal_launch(
    PieDriver* driver,
    const PieFrameDesc* frame,
    PieCompletion completion) {
    const int status = pie::driver::validate::frame_desc(frame);
    if (status != PIE_STATUS_OK) { diagnose_frame(frame); return status; }
    const int completion_status =
        pie::driver::validate::completion(completion, false);
    if (completion_status != PIE_STATUS_OK) return completion_status;
    if (driver == nullptr) return PIE_STATUS_INVALID_ARGUMENT;
    try {
        return as_context(driver)->launch(*frame, completion);
    } catch (const std::exception& e) {
        std::cerr << "[pie-driver-metal] launch: " << e.what() << "\n";
        return PIE_STATUS_DRIVER_ERROR;
    } catch (...) {
        std::cerr << "[pie-driver-metal] launch: unknown exception\n";
        return PIE_STATUS_DRIVER_ERROR;
    }
}

extern "C" int32_t pie_metal_copy_kv(
    PieDriver* driver,
    const PieKvCopyDesc* copy,
    PieCompletion completion) {
    const int status = pie::driver::validate::kv_copy_desc(copy);
    if (status != PIE_STATUS_OK) return status;
    const int completion_status =
        pie::driver::validate::completion(completion, true);
    if (completion_status != PIE_STATUS_OK) return completion_status;
    if (driver == nullptr) return PIE_STATUS_INVALID_ARGUMENT;
    try {
        return as_context(driver)->copy_kv(*copy, completion);
    } catch (const std::exception& e) {
        std::cerr << "[pie-driver-metal] copy_kv: " << e.what() << "\n";
        return PIE_STATUS_DRIVER_ERROR;
    } catch (...) {
        std::cerr << "[pie-driver-metal] copy_kv: unknown exception\n";
        return PIE_STATUS_DRIVER_ERROR;
    }
}

extern "C" int32_t pie_metal_copy_state(
    PieDriver* driver,
    const PieStateCopyDesc* copy,
    PieCompletion completion) {
    const int status = pie::driver::validate::state_copy_desc(copy);
    if (status != PIE_STATUS_OK) return status;
    const int completion_status =
        pie::driver::validate::completion(completion, true);
    if (completion_status != PIE_STATUS_OK) return completion_status;
    if (driver == nullptr) return PIE_STATUS_INVALID_ARGUMENT;
    try {
        return as_context(driver)->copy_state(*copy, completion);
    } catch (const std::exception& e) {
        std::cerr << "[pie-driver-metal] copy_state: " << e.what() << "\n";
        return PIE_STATUS_DRIVER_ERROR;
    } catch (...) {
        std::cerr << "[pie-driver-metal] copy_state: unknown exception\n";
        return PIE_STATUS_DRIVER_ERROR;
    }
}

extern "C" int32_t pie_metal_resize_pool(
    PieDriver* driver,
    const PiePoolResizeDesc* resize,
    PieCompletion completion) {
    const int status = pie::driver::validate::pool_resize_desc(resize);
    if (status != PIE_STATUS_OK) return status;
    const int completion_status =
        pie::driver::validate::completion(completion, true);
    if (completion_status != PIE_STATUS_OK) return completion_status;
    if (driver == nullptr) return PIE_STATUS_INVALID_ARGUMENT;
    try {
        return as_context(driver)->resize_pool(*resize, completion);
    } catch (const std::exception& e) {
        std::cerr << "[pie-driver-metal] resize_pool: " << e.what() << "\n";
        return PIE_STATUS_DRIVER_ERROR;
    } catch (...) {
        std::cerr << "[pie-driver-metal] resize_pool: unknown exception\n";
        return PIE_STATUS_DRIVER_ERROR;
    }
}

extern "C" int32_t pie_metal_close_instance(
    PieDriver* driver,
    std::uint64_t instance_id) {
    if (driver == nullptr) return PIE_STATUS_INVALID_ARGUMENT;
    try {
        return as_context(driver)->close_instance(instance_id);
    } catch (const std::exception& e) {
        std::cerr << "[pie-driver-metal] close_instance: " << e.what() << "\n";
        return PIE_STATUS_DRIVER_ERROR;
    } catch (...) {
        std::cerr << "[pie-driver-metal] close_instance: unknown exception\n";
        return PIE_STATUS_DRIVER_ERROR;
    }
}

extern "C" int32_t pie_metal_close_channel(
    PieDriver* driver,
    std::uint64_t channel_id) {
    if (driver == nullptr || channel_id == 0) return PIE_STATUS_INVALID_ARGUMENT;
    try {
        return as_context(driver)->close_channel(channel_id);
    } catch (const std::exception& e) {
        std::cerr << "[pie-driver-metal] close_channel: " << e.what() << "\n";
        return PIE_STATUS_DRIVER_ERROR;
    } catch (...) {
        std::cerr << "[pie-driver-metal] close_channel: unknown exception\n";
        return PIE_STATUS_DRIVER_ERROR;
    }
}

extern "C" void pie_metal_destroy(PieDriver* driver) {
    delete as_context(driver);
}

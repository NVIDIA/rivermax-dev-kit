/*
 * Copyright © 2017-2023 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 */

#ifndef RMAX_APPS_LIB_LIB_SERVICES_UTILS_DEFS_H_
#define RMAX_APPS_LIB_LIB_SERVICES_UTILS_DEFS_H_

#include <cstdint>
#include <string>
#include <vector>
#include <functional>

namespace ral
{
namespace lib
{
namespace services
{

#ifdef __GNUC__
#define likely(condition) __builtin_expect(static_cast<bool>(condition), 1)
#define unlikely(condition) __builtin_expect(static_cast<bool>(condition), 0)
#else
#define likely(condition) (condition)
#define unlikely(condition) (condition)
#endif
#define NOT_IN_USE(a) ((void)(a))
#define align_down_pow2(_n, _alignment) \
    ( (_n) & ~((_alignment) - 1) )
#define align_up_pow2(_n, _alignment) \
    align_down_pow2((_n) + (_alignment) - 1, _alignment)

/**
 * @brief: Hashing Enum class.
 *
 * This class can be used as a functor for class types that requires hashing for
 * STL operations.
 */
struct EnumClassHash
{
template <typename T>
std::size_t operator()(T t) const { return static_cast<std::size_t>(t); }
};

typedef uint8_t byte_t;
typedef byte_t* byte_ptr_t;

enum class StreamType
{
    Video2110_20,
};

enum class VideoScanType
{
    Progressive,
    Interlaced
};

enum class TPMode
{
    TPN,
    TPNL,
    TPW,
};

typedef struct packet_pacing_rate
{
    uint64_t bps;
    uint32_t max_burst_in_packets;
} pp_rate_t;

typedef struct frame_rate
{
    uint16_t num;
    uint16_t denom;
} frame_rate_t;

typedef struct resolution
{
    uint16_t width;
    uint16_t height;
} resolution_t;

/**
 * @brief: Application media related settings.
 *
 * The struct will be used to hold application media parameters required
 * for the application to operate as requested by the user.
 *
 * @note: TODO: Update/remove this struct after adding SDP parser.
 */
typedef struct media_settings
{
    std::string sdp;
    uint32_t media_block_index;
    StreamType stream_type;
    frame_rate_t frame_rate;
    resolution_t resolution;
    VideoScanType video_scan_type;
    TPMode tp_mode;
    size_t sample_rate;
    uint32_t packets_in_frame_field;
    size_t packets_in_line;
    double frame_field_time_interval_ns;
    size_t lines_in_frame_field;
    size_t chunks_in_frame_field;
    size_t frames_fields_in_mem_block;
} media_settings_t;

/**
 * @brief: Application settings.
 *
 * The struct will be used to hold application parameters required
 * for the application to operate as requested by the user.
 */
typedef struct AppSettings
{
    int sleep_between_operations_us;
    bool sleep_between_operations;
    std::vector<std::string> local_ips;
    std::string source_ip;
    std::vector<std::string> source_ips;
    uint16_t source_port;
    std::string destination_ip;
    std::vector<std::string> destination_ips;
    uint16_t destination_port;
    std::vector<uint16_t> destination_ports;
    uint16_t num_of_total_streams;
    size_t num_of_total_flows;
    size_t num_of_threads;
    int internal_thread_core;
    std::vector<int> app_threads_cores;
    bool print_parameters;
    pp_rate_t rate;
    size_t num_of_memory_blocks;
    size_t num_of_chunks_in_mem_block;
    size_t num_of_chunks;
    size_t num_of_packets_in_chunk;
    size_t num_of_packets_in_mem_block;
    uint16_t packet_payload_size;
    uint16_t packet_app_header_size;
    bool use_checksum_header;
    uint32_t hw_queue_full_sleep_us;
    int gpu_id;
    media_settings_t media;
    std::string video_stream_type;  // TODO: Remove this after adding SDP parser.
} AppSettings;

/**
 * @brief: Input stream settings.
 */
typedef struct receive_stream_settings
{
    uint16_t packet_payload_size = 0;
    uint16_t packet_app_header_size = 0;
    uint32_t num_of_packets_in_chunk = 0;
    size_t min_chunk_size = 0;
    size_t max_chunk_size = 0;
    int timeout_us = 0;
    rmax_in_buffer_attr_flags buffer_attr_flags = RMAX_IN_BUFFER_ATTER_FLAG_NONE;
    rmax_in_flags stream_flags = RMAX_IN_FLAG_NONE;

    receive_stream_settings()
    {
    }

    receive_stream_settings(
            uint16_t packet_payload_size_,
            uint16_t packet_app_header_size_,
            uint32_t num_of_packets_in_chunk_,
            size_t min_chunk_size_,
            size_t max_chunk_size_,
            int timeout_us_,
            rmax_in_buffer_attr_flags buffer_attr_flags_,
            rmax_in_flags stream_flags_
    ) :
        packet_payload_size(packet_payload_size_),
        packet_app_header_size(packet_app_header_size_),
        num_of_packets_in_chunk(num_of_packets_in_chunk_),
        min_chunk_size(min_chunk_size_),
        max_chunk_size(max_chunk_size_),
        timeout_us(timeout_us_),
        buffer_attr_flags(buffer_attr_flags_),
        stream_flags(stream_flags_)
    {
    }
} receive_stream_settings_t;

/**
 * @brief: Redundant stream settings.
 */
typedef struct ipo_stream_settings
{
    uint16_t packet_payload_size = 0;
    uint16_t packet_app_header_size = 0;
    size_t num_of_packets_in_chunk = 0;
    size_t max_chunk_size = 0;
    rmax_in_buffer_attr_flags buffer_attr_flags = RMAX_IN_BUFFER_ATTER_FLAG_NONE;
    rmax_in_flags stream_flags = RMAX_IN_FLAG_NONE;
    uint64_t max_path_differential_us = 0;

    ipo_stream_settings()
    {
    }

    ipo_stream_settings(
            uint16_t packet_payload_size_,
            uint16_t packet_app_header_size_,
            size_t num_of_packets_in_chunk_,
            size_t max_chunk_size_,
            rmax_in_buffer_attr_flags buffer_attr_flags_,
            rmax_in_flags stream_flags_,
            uint64_t max_path_differential_us_
    ) :
        packet_payload_size(packet_payload_size_),
        packet_app_header_size(packet_app_header_size_),
        num_of_packets_in_chunk(num_of_packets_in_chunk_),
        max_chunk_size(max_chunk_size_),
        buffer_attr_flags(buffer_attr_flags_),
        stream_flags(stream_flags_),
        max_path_differential_us(max_path_differential_us_)
    {
    }

} ipo_stream_settings_t;

/**
 * @brief: Time handler callback definition.
 *
 * This defines the callback type for time handling callback functions.
 */
typedef std::function<uint64_t(void*)> time_handler_ns_cb_t;

} // namespace services
} // namespace lib
} // namespace ral

#endif /* RMAX_APPS_LIB_LIB_SERVICES_UTILS_DEFS_H_ */

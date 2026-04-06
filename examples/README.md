# Rivermax Dev Kit Examples

## Overview

The Rivermax Dev Kit includes a set of examples that demonstrate how to use its various modules and APIs. These examples serve as practical tutorials and reference implementations for developers looking to integrate the Rivermax Dev Kit into their applications, covering different aspects of the development kit and showcasing various use cases.

## Structure

The examples directory mirrors the source code architecture of the Rivermax Dev Kit, enabling developers to intuitively navigate and understand example implementations in direct correspondence with the main project structure. This architectural alignment helps developers quickly locate relevant examples and understand how they relate to the underlying framework components.

Each example is designed for clarity and ease of understanding, with the main application flow implemented in the overridden call operator function (`operator()()`), while initialization and configuration details are encapsulated in dedicated functions for readability.

At the top level of the `examples/` directory, you will currently find the following main components:

- **Base Example Framework**: Infrastructure providing standardized initialization, CLI argument parsing, and error handling for all examples. Not required to understand the examples themselves.
- **Apps Module**: Demonstrates end-to-end implementations showcasing real-world usage scenarios and workflows when using the app top-level Rivermax Dev Kit APIs.
- **Services Module**: Focused examples highlighting specific functionality and features of the Rivermax Dev Kit services.

Module directories may contain an `integration/` subdirectory for examples that coordinate multiple modules within that category to achieve unified workflows.

All examples follow the same namespace structure as the main project, using `rdk::examples` as the prefix. Integration examples add an `integration` namespace level (e.g., `rdk::examples::apps::integration`).

Additional examples at different levels will be added in the future to cover more modules, features and use cases of the Rivermax Dev Kit.

## Examples Description

### Apps Module

The `apps/` directory contains application level examples that demonstrate real-world usage scenarios and end-to-end workflows.

#### rdk_media_sender

This submodule contains examples for media streaming:

- **cli_settings_media_file_sender**: This example demonstrates CLI-based media file streaming with configurable command-line parameters. It shows how to implement file-based media transmission using user-configurable options and demonstrates proper command-line parameter parsing and handling.

- **user_provided_settings_video_file_sender**: This example demonstrates software-based video file transmission using app-level settings structures for configuration. It shows how to manage configuration through dedicated settings objects and implement programmatic configuration for file-based video streaming.

- **video_frames_sender**: This example demonstrates video frame transmission using the media unit abstract API for efficient media streaming. It shows how to use the media essence source API where media units represent video frames, providing direct control over individual frame processing and delivery.

#### rdk_rtp_receiver

This submodule contains examples for RTP packet reception and processing:

- **rtp_chunks_receiver**: This example demonstrates RTP stream reception and packet processing using chunk-based APIs. It shows how to configure app-level APIs for RTP stream reception and utilize chunk-based processing methods to efficiently receive and handle RTP packets.

#### Integration Examples

The `integration/` subdirectory contains examples that demonstrate how to combine multiple app-level modules to create unified workflows:

- **media_receiver_sender**: This example demonstrates bidirectional media streaming by implementing a combined receiver-sender application. It shows how to integrate Rivermax Dev Kit app-level APIs for both receiving and transmitting media streams in a single application using the RTP receiver and media sender modules.

### Services Module

The `services/` directory contains focused examples for specific Rivermax Dev Kit services.

#### memory_allocation

This submodule contains examples for memory allocation strategies:

- **memory_allocation**: This example demonstrates how to use different memory allocation strategies provided by the Rivermax Dev Kit. It shows how to allocate memory using HugePage allocators for optimized memory access and GPU allocators for GPU-based operations. The example includes optional GPU allocation that can be controlled via command-line parameters, demonstrating how to handle different allocation types based on system capabilities.

#### sdp

This submodule contains examples for SDP generation:

- **sdp_smpte_2110_20_description**: This example demonstrates how to generate Session Description Protocol (SDP) strings for SMPTE ST 2110-20 video streams using the SDP service.

- **sdp_smpte_2110_30_description**: This example demonstrates how to generate Session Description Protocol (SDP) strings for SMPTE ST 2110-30 audio streams using the SDP service.

- **sdp_smpte_2110_40_description**: This example demonstrates how to generate Session Description Protocol (SDP) strings for SMPTE ST 2110-40 ancillary data streams using the SDP service.

- **sdp_smpte_2110_mixed_description**: This example demonstrates how to generate Session Description Protocol (SDP) strings for combined SMPTE ST 2110-20 video, ST 2110-30 audio, and ST 2110-40 ancillary data streams.

#### ulp_packet_buffer

This submodule contains examples for ULP packet buffer writing functionality:

- **rtp_smpte_2110_20_packet_buffer_writer**: This example demonstrates SMPTE 2110-20 compliant RTP packet buffer writing functionality. It shows how to properly configure and use RTP packet buffer writers to handle SMPTE 2110-20 specific requirements.

## Building and Running Examples

### Building

The examples are built automatically as part of the main Rivermax Dev Kit build process. In addition, one can build the examples individually by configuring and building the examples target using the following steps:

1. First, set up the build environment by running the following command in the root directory of the Rivermax Dev Kit:
```sh
cmake -B <build-dir> -DCMAKE_BUILD_TYPE=Release
```
2. Next, build the examples by running:
```sh
cmake --build <build-dir> --config Release --parallel --target rivermax-dev-kit-examples
```
3. After the building process completes, you can find the example executables in the `<build-dir>/examples` directory, organized by their respective submodules, following the same structure as the source code.

### Running Examples

To run an example:

```sh
cd <build-dir>/examples/<example-path>
./example_name --help  # Display usage information
./example_name [options]  # Run with specific options
```

Each example includes predefined variables that can be seen in the code. Some examples may require additional configuration files or command-line arguments to function correctly.

## Contributing

### Base Example Framework

All examples inherit from `BaseExample` (`rdk::examples::BaseExample`) which provides:

- Standardized initialization and settings management
- Built-in CLI argument parsing with automatic help generation
- Consistent error handling and reporting
- Unified configuration system

All example-related code should be implemented in the overridden call operator function (`operator()()`).

### Example Placement

Examples should be organized to mirror the source code structure and clearly reflect their purpose:

- **Module-specific examples**: Place in the corresponding module's directory (e.g., `apps/rdk_rtp_receiver/`, `services/memory_allocation/`). Use this for examples that demonstrate features, capabilities, or workflows of a single module, even if they use other modules as dependencies or supporting functionality.

- **Integration examples**: Place in an `integration/` subdirectory within the module category (e.g., `apps/integration/`, `services/integration/`). Use this only when the example's primary purpose is demonstrating coordination between multiple modules, where the modules are used equally and centrally to achieve the example's unified workflow.

When in doubt, prefer module-specific placement unless the example truly requires equal focus on multiple modules.

### Development Guidelines

1. **Inherit from BaseExample** for consistency
2. **Use descriptive naming** that reflects functionality
3. **Follow existing code style and patterns**
4. **Include comprehensive documentation** and usage instructions
5. **Implement error handling** with meaningful messages
6. **Test across the affected use cases**

### Integration

Examples are integrated with the main project through:

- **Namespace**: `rdk::examples` with sub-namespaces following the source structure (e.g., `rdk::examples::services`, `rdk::examples::apps`)
- **Dependencies**: Link against Rivermax Dev Kit targets
- **Build System**: Integrated with main CMake build
- **Documentation**: Included in generated API docs

### Contribution Process

1. **Review existing examples** to understand patterns
2. **Implement following guidelines** above
3. **Test thoroughly** across affected use cases
4. **Update documentation** including this README

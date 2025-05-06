/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "rdk/apps/rmax_base_app.h"
#include "rdk/apps/rmax_latency/rmax_latency.h"

using namespace rivermax::dev_kit::apps;
using namespace rivermax::dev_kit::apps::rmax_latency;

/**
 * Application constants.
 */
static constexpr const char* APP_DESCRIPTION = "NVIDIA Rivermax latency measurement application";

static constexpr const char* APP_USAGE_MESSAGE =
R"EOF(The latency measurement tool implements the following measurement modes:
  Ping-Pong mode - sending a single packet from Client to Server and back, using Rivermax Tx Generic API.
    In this mode packet round-trip time is measured, as well as the following internal Rivermax latencies:
       - Between chunk commit time and the time received in the HW Tx completion
       - Between the time received in the HW Rx completion and the time the SW received the next chunk.
  Frame mode - sending an array of data (representing a video frame or segment)
               from Client to Server using Rivermax Tx Generic API with max speed, without packet pacing.
    In this mode the total transmission time is measured from commit time of the first chunk in frame by the Client SW
    to the time of receiving the last chunk in frame by the Server SW.
    In this mode the application also shows the following intervals:
    - Transmit start latency:         from the time of the first SW commit to the time provided by the first Tx completion
    - Sending time of entire frame:   from the time of the first SW commit to the time provided by the last Tx completion
    - Receiving time of entire frame: from the time provided by the first HW Rx completion to the time when the last chunk was received by the SW
    - Receive latency:                from the time provided by the last HW Rx completion to the time of the last SW chunk reception
    - Reply end-to-end latency:       from the time of the reply chunk commit by the server SW to its reception by the client SW
  Media mode - sending a SMPTE ST2110-20 video stream from Client to Server.
    In this mode the following delays are measured:
    - On Client side: between a packet HW transmission time and its scheduled packet consumption time
                      (TPRj as specified by the SMPTE ST2110-21 Gapped PRS model)
    - On Server side: between a packet HW reception time and its scheduled packet consumption time
                      (TPRj as specified by the SMPTE ST2110-21 Gapped PRS model)
For measurements involving HW timestamps, PTP-synchronized HW Real Time Clock must be enabled in the NIC (see Rivermax user manual).

The following parameters are available in all measurement modes:
   -l,--local-ip       local IP of the NIC
   -d,--dst-ip         send stream destination IP
   -p,--dst-port       send stream destination port
   -r,--receive-ip     receive stream destination IP
   -o,--receive-port   receive stream destination port
   --mode              measurement mode (pp, frame, media)
   -c,--client         operate as client, if not specified - operate as server
On the client side the measurement interval can be specified:
   -m,--measure        measurement interval, sec
Configurable parameters for ping-pong mode:
   -y,--payload-size   packet size
   -A,--allocator-type memory allocator type
   -z,--sleep-us       pause between requests (usec)
Configurable parameters for frame mode:
   -y,--payload-size   packet size
   -A,--allocator-type memory allocator type (defaults to GPU if GPUDirect is used)
   -g,--gpu-id         GPU id for GPUDirect
   -z,--sleep-us       pause between requests (usec)
   -C,--chunks         number of chunks in frame
   -K,--packets        number of packets in chunk
Configurable parameters for media mode:
   -A,--allocator-type memory allocator type (defaults to GPU if GPUDirect is used)
   -g,--gpu-id         GPU id for GPUDirect
   -K,--packets        number of packets in chunk
Video format options:
  --vr,--video-resolution TEXT:Video resolution validator:{1920x1080,3840x2160}=1920x1080
                              Video resolution in format <width>x<height>
  --vfr,--video-frame-rate TEXT:Video frame rate validator:{50,60}=60
                              Video frame rate in format <numerator>/<denominator> or <integer>

Examples:
  1. Ping-Pong mode
  client: rmax_latency -l 1.2.3.4 -d 224.2.3.4 -p 2000 -r 224.2.3.5 -o 2000 -i1 -a2 -c -m10
  server: rmax_latency -l 1.2.3.5 -d 224.2.3.5 -p 2000 -r 224.2.3.4 -o 2000 -i1 -a2
  2. Frame Latency mode
  client: rmax_latency -l 1.2.3.4 -d 224.2.3.4 -p 2000 -r 224.2.3.5 -o 2000 -i1 -a2 -c -m10 --mode frame -C100 -K20
  server: rmax_latency -l 1.2.3.5 -d 224.2.3.5 -p 2000 -r 224.2.3.4 -o 2000 -i1 -a2 --mode frame -C100 -K20
  3. Media Latency mode
  client: rmax_latency -l 1.2.3.4 -d 224.2.3.4 -p 2000 -r 224.2.3.5 -o 2000 -i1 -a2 -c -m10 --mode media -x 1080p60 -K4
  server: rmax_latency -l 1.2.3.5 -d 224.2.3.5 -p 2000 -r 224.2.3.4 -o 2000 -i1 -a2 --mode media -x 1080p60;
)EOF";

int main(int argc, const char* argv[])
{
    return common_cli_main<LatencyApp, LatencyCLISettingsBuilder, LatencySettingsValidator>(
        argc, argv, APP_DESCRIPTION, APP_USAGE_MESSAGE);
}

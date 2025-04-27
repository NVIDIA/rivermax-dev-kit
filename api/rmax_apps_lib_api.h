/*
 * Copyright © 2021-2024 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 */

#ifndef RMAX_APPS_LIB_API_RMAX_APPS_LIB_API_H_
/*
* @ Brief: Main API of Rivermax application library.
*
* This API include file, includes all library packages include files.
* Each package include file, contains the relevant API include files.
* In order to see the relevant class interface:
*     1. Go to the relevant package include file, e.g. "lib/core/stream/stream.h"
*     2. In the chosen package include file, go to the relevant interface include file
*        e.g. In the file above, in order to see the stream interface, go to: "core/stream/stream_interface.h"
*
* Future packages should follow the same convention, the generic structure presented below.
*
* Generic structure:
*
* |   ...
* |-- lib
*   |-- core
*   |   ...
*   |   |-- future_core_package_x
*   |   |   |-- new_core_interface_a.h
*   |   |   |-- new_core_interface_a.cpp
*   |   |   ...
*   |   |   |-- future_core_package_x.h
*   |-- services
*   |    ...
*   |    |-- future_service_package_y
*   |        |-- new_service_interface_a.h
*   |        |-- new_service_interface_a.cpp
*   |        ...
*   |        |-- future_service_package_y.h
*   |-- rmax_apps_lib_facade.h
*
* CPP namespace structure:
*
* All core components of the library reside in namespace ral::lib::core.
* All services components of the library reside in namespace ral::lib::services.
*
*/
#include "lib/services/cli/cli.h"
#include "lib/services/utils/utils.h"
#include "lib/services/error_handling/error_handling.h"
#include "lib/services/memory_management/memory_management.h"
#include "lib/services/buffer_management/buffer_management.h"
#include "lib/services/statistics/statistics_reader.h"
#include "lib/core/stream/stream.h"
#include "lib/core/flow/flow.h"
#include "lib/core/chunk/chunk.h"
#include "lib/rmax_apps_lib_facade.h"

#define RMAX_APPS_LIB_API_RMAX_APPS_LIB_API_H_
#endif /* RMAX_APPS_LIB_API_RMAX_APPS_LIB_API_H_ */

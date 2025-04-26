lib_LTLIBRARIES = librmaxapps.la

librmaxapps_la_CXXFLAGS = -fPIC
librmaxapps_la_LDFLAGS = -ldl -lpthread
librmaxapps_la_LIBADD = $(EXTERNAL_LIBS)

librmaxapps_la_CPPFLAGS = \
    -I$(EXTERNAL_API) \
    -I$(EXTERNAL_SRCDIR) \
    -I$(top_srcdir) \
    -I$(top_srcdir)/lib

# This workaround allows to compile files located
# at another directory.
# This place resolve make distcheck issue
nodist_librmaxapps_la_SOURCES = \
    rt_threads.cpp \
    gpu.cpp

CLEANFILES = = \
    rt_threads.cpp \
    gpu.cpp

rt_threads.cpp:
	@echo "#include \"$(EXTERNAL_SRCDIR)/$@\"" >$@

gpu.cpp:
	@echo "#include \"$(EXTERNAL_SRCDIR)/$@\"" >$@

librmaxapps_la_SOURCES = \
    \
    services/utils/utils.h \
    services/utils/defs.h \
    \
    services/error_handling/error_handling.h \
    services/error_handling/return_status.h \
    services/error_handling/exceptions.h \
    services/error_handling/signal_handler.h \
    services/error_handling/signal_handler.cpp \
    \
    services/cli/cli.h \
    services/cli/cli_manager.h \
    services/cli/cli_manager.cpp \
    services/cli/validators.h \
    services/cli/validators.cpp \
    services/cli/options.h \
    services/cli/options.cpp \
    \
    services/memory_management/memory_management.h \
    services/memory_management/memory_allocator_interface.h \
    services/memory_management/memory_allocator_interface.cpp \
    services/memory_management/new_memory_allocator.h \
    services/memory_management/new_memory_allocator.cpp \
    services/memory_management/gpu_memory_allocator.h \
    services/memory_management/gpu_memory_allocator.cpp \
    \
    services/buffer_management/buffer_management.h \
    services/buffer_management/buffer_writer_interface.h \
    services/buffer_management/generic_buffer_writer.h \
    services/buffer_management/generic_buffer_writer.cpp \
    services/buffer_management/checksum_buffer_writer.h \
    services/buffer_management/checksum_buffer_writer.cpp \
    \
    core/flow/flow.h \
    core/flow/flow.cpp \
    \
    core/chunk/chunk.h \
    core/chunk/chunk_interface.h \
    core/chunk/generic_chunk.h \
    core/chunk/generic_chunk.cpp \
    core/chunk/receive_chunk.h \
    core/chunk/receive_chunk.cpp \
    core/chunk/media_chunk.h \
    core/chunk/media_chunk.cpp \
    \
    core/stream/stream.h \
    core/stream/stream_interface.h \
    core/stream/stream_interface.cpp \
    core/stream/single_stream_interface.h \
    core/stream/single_stream_interface.cpp \
    core/stream/aggregate_stream_interface.h \
    core/stream/aggregate_stream_interface.cpp \
    core/stream/send/send_stream_interface.h \
    core/stream/send/send_stream_interface.cpp \
    core/stream/send/generic_stream.h \
    core/stream/send/generic_stream.cpp \
    core/stream/send/media_stream.h \
    core/stream/send/media_stream.cpp \
    core/stream/receive/ipo_receive_stream.h \
    core/stream/receive/ipo_receive_stream.cpp \
    core/stream/receive/receive_stream.h \
    core/stream/receive/receive_stream.cpp \
    \
    rmax_apps_lib_facade.h \
    rmax_apps_lib_facade.cpp

otherincludedir = $(includedir)/mellanox
otherinclude_HEADERS = $(top_srcdir)/api/rmax_apps_lib_api.h

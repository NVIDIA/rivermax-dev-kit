lib_LTLIBRARIES = libionode.la

libionode_la_CPPFLAGS = \
    -I$(EXTERNAL_API) \
    -I$(EXTERNAL_SRCDIR) \
    -I$(top_srcdir) \
    -I$(top_srcdir)/lib

libionode_la_LDFLAGS = \
    -static

libionode_la_LIBADD = \
    $(EXTERNAL_LIBS) \
    $(top_builddir)/lib/librmaxapps.la

libionode_la_DEPENDENCIES = \
    $(top_builddir)/lib/librmaxapps.la

libionode_la_SOURCES = \
    io_node.h \
    senders/generic_sender_io_node.h \
    senders/generic_sender_io_node.cpp \
    senders/media_sender_io_node.h \
    senders/media_sender_io_node.cpp \
    receivers/ipo_receiver_io_node.h \
    receivers/ipo_receiver_io_node.cpp

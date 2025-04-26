noinst_PROGRAMS = rmax_xstream_media_sender

rmax_xstream_media_sender_CPPFLAGS = \
    -I$(EXTERNAL_API) \
    -I$(EXTERNAL_SRCDIR) \
    -I$(top_srcdir) \
    -I$(top_srcdir)/lib

rmax_xstream_media_sender_LDFLAGS = \
    -pthread -no-install

rmax_xstream_media_sender_LDADD = \
    $(EXTERNAL_LIBS) \
    $(top_builddir)/io_node/libionode.la \
    $(top_builddir)/apps/librmaxbaseapp.la

rmax_xstream_media_sender_DEPENDENCIES = \
    $(top_builddir)/io_node/libionode.la \
    $(top_builddir)/apps/librmaxbaseapp.la

rmax_xstream_media_sender_SOURCES = \
    rmax_xstream_media_sender.h \
    rmax_xstream_media_sender.cpp

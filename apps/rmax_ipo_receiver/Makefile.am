noinst_PROGRAMS = rmax_ipo_receiver

rmax_ipo_receiver_CPPFLAGS = \
    -I$(EXTERNAL_API) \
    -I$(EXTERNAL_SRCDIR) \
    -I$(top_srcdir) \
    -I$(top_srcdir)/lib

rmax_ipo_receiver_LDFLAGS = \
    -pthread -no-install

rmax_ipo_receiver_LDADD = \
    $(EXTERNAL_LIBS) \
    $(top_builddir)/io_node/libionode.la \
    $(top_builddir)/apps/librmaxbaseapp.la

rmax_ipo_receiver_DEPENDENCIES = \
    $(top_builddir)/io_node/libionode.la \
    $(top_builddir)/apps/librmaxbaseapp.la

rmax_ipo_receiver_SOURCES = \
    rmax_ipo_receiver.h \
    rmax_ipo_receiver.cpp

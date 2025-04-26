noinst_LTLIBRARIES = librmaxbaseapp.la

librmaxbaseapp_la_CPPFLAGS = \
    -I$(EXTERNAL_API) \
    -I$(EXTERNAL_SRCDIR) \
    -I$(top_srcdir) \
    -I$(top_srcdir)/lib

librmaxbaseapp_la_LDFLAGS = \
    -static

librmaxbaseapp_la_LIBADD = \
    $(EXTERNAL_LIBS) \
    $(top_builddir)/lib/librmaxapps.la

librmaxbaseapp_la_DEPENDENCIES = \
    $(top_builddir)/lib/librmaxapps.la

librmaxbaseapp_la_SOURCES = \
    rmax_base_app.h \
    rmax_base_app.cpp

# Contains the subdirectories of the current directory that
# must be built
SUBDIRS = \
	lib \
	io_node \
	apps \
	apps/rmax_xstream_generic_sender \
	apps/rmax_xstream_media_sender \
	apps/rmax_ipo_receiver

if SUPPORT_EXTERNAL
EXTERNAL_SUBDIR = util
else
EXTERNAL_SUBDIR =
endif

EXTRA_DIST = \
	autogen.sh \
	$(EXTERNAL_SUBDIR)

.PHONY: lib io_node apps docs

lib:
	$(MAKE) -C lib

io_node: lib
	$(MAKE) -C io_node

apps: io_node
	$(MAKE) -C apps
	$(MAKE) -C apps/rmax_xstream_generic_sender
	$(MAKE) -C apps/rmax_xstream_media_sender
	$(MAKE) -C apps/rmax_ipo_receiver

if HAVE_DOXYGEN
if SUPPORT_EXTERNAL
docs:
	$(info docs generation available in development build only)
else
# This target should be run from rmax_apps_lib folder.
docs:
	$(DOXYGEN) $(top_srcdir)/docs/doxygen.config.in

clean-local:
	-rm -rf $(top_srcdir)/docs/html
	-rm -f $(top_srcdir)/docs/doxygen.config
endif
endif

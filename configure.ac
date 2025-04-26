# Indicate that we require autoconf 2.63 or later
#
AC_PREREQ(2.63)


dnl===-----------------------------------------------------------------------===
dnl===
dnl=== SECTION 1: Setting product version
dnl===
dnl===-----------------------------------------------------------------------===

# Update version number here:
#
define([prj_ver_major], 0)
define([prj_ver_minor], 1)
define([prj_ver_revision], 0)
define([prj_ver_release], esyscmd([echo ${PRJ_RELEASE:=0}]))


# Initialize autoconf and define the package name, version number and
# email address for reporting bugs.
#
AC_INIT(rmax_apps_lib, [prj_ver_major.prj_ver_minor.prj_ver_revision], support@mellanox.com)

# Definitions will be placed in this file rather than
# in the DEFS variable
#
AC_CONFIG_HEADER([config.h])

PRJ_MAJOR=prj_ver_major
PRJ_MINOR=prj_ver_minor
PRJ_REVISION=prj_ver_revision
PRJ_RELEASE=prj_ver_release

AC_DEFINE_UNQUOTED(PRJ_MAJOR, ${PRJ_MAJOR}, [Project Major Version])
AC_SUBST(PRJ_MAJOR)
AC_DEFINE_UNQUOTED(PRJ_MINOR, ${PRJ_MINOR}, [Project Minor Version])
AC_SUBST(PRJ_MINOR)
AC_DEFINE_UNQUOTED(PRJ_REVISION, ${PRJ_REVISION}, [Project Revision])
AC_SUBST(PRJ_REVISION)
AC_DEFINE_UNQUOTED(PRJ_RELEASE, ${PRJ_RELEASE}, [Project Release])
AC_SUBST(PRJ_RELEASE)

BUILD_VER=`git describe --long --abbrev=40 --dirty --tags 2> /dev/null || echo ""`
if test -n "$BUILD_VER"; then BUILD_VER=`echo $BUILD_VER | sed -e 's/-dirty/+/' | sed s/.*-g//`; else  BUILD_VER=""; fi

AC_DEFINE_UNQUOTED(BUILD_VER, "${BUILD_VER}", [SCM Version])
AC_SUBST(BUILD_VER)

dateopt=""
    if test -n "$SOURCE_DATE_EPOCH" ; then
    dateopt="-u -d @$SOURCE_DATE_EPOCH"
fi
AC_SUBST([BUILD_DATE], [$(date $dateopt +'%b/%d/%Y')])
AC_SUBST([BUILD_TIME], [$(date $dateopt +'%H:%M:%S')])
AC_SUBST([BUILD_DATE_CHANGELOG], [$(date $dateopt +'%a, %d %b %Y %T %z')])

##########################
# Configure functions
#
# Some helper script functions
#
AC_DEFUN([FUNC_CONFIGURE_INIT],
[
show_section_title()
{
    cat <<EOF

============================================================================
== ${1}
============================================================================
EOF
}

show_summary_title()
{
    cat <<EOF

${PACKAGE_NAME}
============================================================================
Version: ${PACKAGE_VERSION}
Build: ${BUILD_VER}

EOF
}

])

# FUNC_CHECK_WITHDIR(name, direcory, file)
# ----------------------------------------------------
AC_DEFUN([FUNC_CHECK_WITHDIR],[
    AC_MSG_CHECKING([for $1 location])
    AS_IF([test "$2" = "yes" || test "$2" = "no" || test "x$2" = "x"],
          [AC_MSG_RESULT([(system default)])],
          [AS_IF([test ! -d "$2"],
                 [AC_MSG_RESULT([not found])
                  AC_MSG_WARN([Directory $2 not found])
                  AC_MSG_ERROR([Cannot continue])],
                 [AS_IF([test "x`ls $2/$3 2> /dev/null`" = "x"],
                        [AC_MSG_RESULT([not found])
                         AC_MSG_WARN([Expected file $2/$3 not found])
                         AC_MSG_ERROR([Cannot continue])],
                        [AC_MSG_RESULT([($2)])]
                       )
                 ]
                )
          ]
         )
])

dnl===-----------------------------------------------------------------------===
dnl===
dnl=== SECTION 2: Initialization & Setup
dnl===
dnl===-----------------------------------------------------------------------===

# Verify that the source directory is valid.
#
AC_CONFIG_SRCDIR(api)

# Init automake and libtool
#
AM_INIT_AUTOMAKE(foreign [subdir-objects])

# Set non-verbose make by default
#
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])

FUNC_CONFIGURE_INIT()


dnl===-----------------------------------------------------------------------===
dnl===
dnl=== SECTION 3: Checking for programs we need
dnl===
dnl===-----------------------------------------------------------------------===
show_section_title "Configure build tools"

# Find compiler, libtools, etc
#
AC_PROG_CC
AC_PROG_CXX
AC_PROG_LIBTOOL

# Check for pkg-config package
#
AC_CHECK_PROG(
    [have_pkg_config],
    [pkg-config],
    m4_ifdef([PKG_PROG_PKG_CONFIG], yes, no_pkg_m4),
    no_pkg_config)
case "$have_pkg_config" in
    no_pkg_m4)
        AC_MSG_ERROR([
        *** You do not have pkg.m4 properly installed.
        *** aclocal can not find one.
        *** Set the environment variable ACLOCAL="aclocal -I/path/to/pkg.m4"
        ])
        ;;
    no_pkg_config)
        AC_MSG_ERROR([
        *** The pkg-config is required to build the library.
        *** Make sure it is installed or set path to pkg-config.
        ])
        ;;
    *)
        ;;
esac
PKG_PROG_PKG_CONFIG()

AC_CHECK_PROGS([DOXYGEN], [doxygen])
if test -z "$DOXYGEN"; then
    AC_MSG_WARN([doxygen not found - continue without doxygen support])
fi
AC_CHECK_PROGS([DOT], [dot])
if test -z "$DOT"; then
    AC_MSG_WARN([doxygen needs dot, please install dot from graphviz package first])
fi
AM_CONDITIONAL([HAVE_DOXYGEN], [test -n "$DOXYGEN"])
AM_CONDITIONAL([HAVE_DOXYGEN], [test -n "$DOT"])
AM_COND_IF([HAVE_DOXYGEN], [AC_CONFIG_FILES([docs/doxygen.config])])

dnl===-----------------------------------------------------------------------===
dnl===
dnl=== SECTION 4: Setting compiler specific options
dnl===
dnl===-----------------------------------------------------------------------===
show_section_title "Setting compiler specific options"

CFLAGS="-D_GNU_SOURCE -fPIC $CFLAGS"
CXXFLAGS="-D_GNU_SOURCE -fPIC $CXXFLAGS"

AC_MSG_CHECKING([for compiler])
case $CC in
    gcc*|g++*)
        AC_MSG_RESULT([gcc])
        CFLAGS="$CFLAGS -Wall -Wextra -Werror -Wno-unknown-pragmas -ffunction-sections -fdata-sections -pipe -Wmissing-include-dirs -Wunused-local-typedefs -Wvla"
        CXXFLAGS="$CXXFLAGS -Wall -Wextra -Werror -Wno-unknown-pragmas -ffunction-sections -fdata-sections -pipe -Wmissing-include-dirs -Wno-overloaded-virtual -Wnon-virtual-dtor -Woverloaded-virtual -Wunused-local-typedefs -Wvla"
        ;;
    clang*|clang++*)
        AC_MSG_RESULT([clang])
        CFLAGS="$CFLAGS -Wall -Werror"
        CXXFLAGS="$CXXFLAGS -Wall -Werror"
        ;;
    *)
        AC_MSG_RESULT([unknown])
        ;;
esac

CXXFLAGS="$CXXFLAGS -std=c++11"


dnl===-----------------------------------------------------------------------===
dnl===
dnl=== SECTION 5: Checking for project configuration
dnl===
dnl===-----------------------------------------------------------------------===
show_section_title "Configure project"


dnl===-----------------------------------------------------------------------===
dnl===
dnl=== SECTION 6: Checking for header files
dnl===
dnl===-----------------------------------------------------------------------===
show_section_title "Check for header files"


dnl===-----------------------------------------------------------------------===
dnl===
dnl=== SECTION 7: Checking for libraries
dnl===
dnl===-----------------------------------------------------------------------===
show_section_title "Check for libraries"

# librivermax usage support
#
AC_ARG_WITH([rivermax],
    AS_HELP_STRING([--with-rivermax(=DIR)],
                   [Search for rivermax headers and libraries in DIR (default YES)]),
    [],
    [with_rivermax=yes]
)

AC_ARG_WITH([rivermax_blddir],
    AS_HELP_STRING([--with-rivermax-blddir(=DIR)],
                   [Search for rivermax build directory in DIR (default NO)]),
    [],
    [with_rivermax_blddir=no]
)

AC_ARG_WITH([rivermax-srcdir],
    AS_HELP_STRING([--with-rivermax-srcdir(=DIR)],
                   [Search for rivermax source directory in DIR (default NO)]),
    [],
    [with_rivermax_srcdir=no]
)

prj_cv_rivermax=0
AS_IF([test "x$with_rivermax" == xno],
    [AS_IF([test "x$with_rivermax_blddir" != xno -a "x$with_rivermax_srcdir" != xno],
        [prj_cv_rivermax=2],
        [AC_MSG_ERROR([rivermax support is mandatory])])
    ],
    [
    if test ! -d "${ac_abs_confdir}/util"; then
        AC_MSG_ERROR([Directory ${ac_abs_confdir}/util does not exist])
    fi

    if test -z "$with_rivermax" || test "$with_rivermax" = "yes"; then
        with_rivermax=/usr
    fi

    FUNC_CHECK_WITHDIR([rivermax], [$with_rivermax], [include/mellanox/rivermax_api.h])

    prj_cv_rivermax_save_CPPFLAGS="$CPPFLAGS"
    prj_cv_rivermax_save_CXXFLAGS="$CXXFLAGS"
    prj_cv_rivermax_save_CFLAGS="$CFLAGS"
    prj_cv_rivermax_save_LDFLAGS="$LDFLAGS"
    prj_cv_rivermax_save_LIBS="$LIBS"

    prj_cv_rivermax_CPPFLAGS="-I$with_rivermax/include -I$with_rivermax/include/mellanox"
    prj_cv_rivermax_LIBS="-lrivermax -lmlx5"
    prj_cv_rivermax_LDFLAGS="-L$with_rivermax/lib -Wl,--rpath,$with_rivermax/lib"
    if test -d "$with_rivermax/lib64"; then
        prj_cv_rivermax_LDFLAGS="-L$with_rivermax/lib64 -Wl,--rpath,$with_rivermax/lib64"
    fi

    CPPFLAGS="$prj_cv_rivermax_CPPFLAGS $CPPFLAGS"
    CXXFLAGS="-std=c++11 $CXXFLAGS"
    LDFLAGS="$prj_cv_rivermax_LDFLAGS $LDFLAGS"
    LIBS="$prj_cv_rivermax_LIBS $LIBS"

    AC_LANG_PUSH([C++])
    AC_CHECK_HEADER(
        [mellanox/rivermax_api.h],
        [AC_LINK_IFELSE([AC_LANG_PROGRAM([[#include <mellanox/rivermax_api.h>]],
             [[const char *rmax_version = rmax_get_version_string(); (void)rmax_version;]])],
             [prj_cv_rivermax=1])
        ])
    AC_LANG_POP()

    CPPFLAGS="$prj_cv_rivermax_save_CPPFLAGS"
    CXXFLAGS="$prj_cv_rivermax_save_CXXFLAGS"
    CFLAGS="$prj_cv_rivermax_save_CFLAGS"
    LDFLAGS="$prj_cv_rivermax_save_LDFLAGS"
    LIBS="$prj_cv_rivermax_save_LIBS"
    ])

AC_MSG_CHECKING([for rivermax support])
if test "$prj_cv_rivermax" -eq 1; then
    CPPFLAGS="$CPPFLAGS $prj_cv_rivermax_CPPFLAGS"
    LDFLAGS="$LDFLAGS $prj_cv_rivermax_LDFLAGS"
    AC_SUBST([EXTERNAL_API], ["${with_rivermax}/include/mellanox"])
    AC_SUBST([EXTERNAL_SRCDIR], ["${ac_abs_confdir}/util"])
    AC_SUBST([EXTERNAL_LIBS], ["-lrivermax -lmlx5"])
    AC_MSG_RESULT([yes])
elif test "$prj_cv_rivermax" -eq 2; then
    AC_SUBST([EXTERNAL_API], ["${with_rivermax_srcdir}/include"])
    AC_SUBST([EXTERNAL_SRCDIR], ["${with_rivermax_srcdir}/tests/util"])
    AC_SUBST([EXTERNAL_LIBS], ["${with_rivermax_blddir}/src/librivermax.la"])
    AC_SUBST([EXTERNAL_RIVERMAX_BLDDIR], ["${with_rivermax_blddir}])
    AC_SUBST([EXTERNAL_RIVERMAX_SRCDIR], ["${with_rivermax_srcdir}])
    AC_MSG_RESULT([yes])
else
    AS_IF([test "x$with_rivermax" == xno],
        [AC_MSG_RESULT([no])],
        [AC_MSG_ERROR([rivermax support requested but not present])])
fi
AM_CONDITIONAL(SUPPORT_EXTERNAL, test "$prj_cv_rivermax" -eq 1)


dnl===-----------------------------------------------------------------------===
dnl===
dnl=== SECTION 8: Checking for types and structures
dnl===
dnl===-----------------------------------------------------------------------===
show_section_title "Check for functions, types and structures"

dnl===-----------------------------------------------------------------------===
dnl===
dnl=== SECTION 9: Configure makefiles
dnl===
dnl===-----------------------------------------------------------------------===
show_section_title "Configure makefiles"
AC_CONFIG_FILES([
		Makefile
		io_node/Makefile
		lib/Makefile
		apps/Makefile
		apps/rmax_xstream_generic_sender/Makefile
		apps/rmax_xstream_media_sender/Makefile
		apps/rmax_ipo_receiver/Makefile
		])

AC_OUTPUT

show_summary_title

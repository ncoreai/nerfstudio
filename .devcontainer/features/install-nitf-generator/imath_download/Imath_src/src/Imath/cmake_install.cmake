# Install script for directory: /workspace/.devcontainer/features/install-nitf-generator/imath_download/Imath_src/src/Imath

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libImath-3_1.so.29.8.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libImath-3_1.so.29"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "\$ORIGIN/../lib:/usr/local/lib")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES
    "/workspace/.devcontainer/features/install-nitf-generator/imath_download/Imath_src/src/Imath/libImath-3_1.so.29.8.0"
    "/workspace/.devcontainer/features/install-nitf-generator/imath_download/Imath_src/src/Imath/libImath-3_1.so.29"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libImath-3_1.so.29.8.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libImath-3_1.so.29"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHANGE
           FILE "${file}"
           OLD_RPATH "::::::::::::::::::::::::::::::"
           NEW_RPATH "\$ORIGIN/../lib:/usr/local/lib")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libImath-3_1.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libImath-3_1.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libImath-3_1.so"
         RPATH "\$ORIGIN/../lib:/usr/local/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/workspace/.devcontainer/features/install-nitf-generator/imath_download/Imath_src/src/Imath/libImath-3_1.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libImath-3_1.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libImath-3_1.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libImath-3_1.so"
         OLD_RPATH "::::::::::::::::::::::::::::::"
         NEW_RPATH "\$ORIGIN/../lib:/usr/local/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libImath-3_1.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/Imath" TYPE FILE FILES
    "/workspace/.devcontainer/features/install-nitf-generator/imath_download/Imath_src/src/Imath/half.h"
    "/workspace/.devcontainer/features/install-nitf-generator/imath_download/Imath_src/src/Imath/halfFunction.h"
    "/workspace/.devcontainer/features/install-nitf-generator/imath_download/Imath_src/src/Imath/halfLimits.h"
    "/workspace/.devcontainer/features/install-nitf-generator/imath_download/Imath_src/src/Imath/ImathBox.h"
    "/workspace/.devcontainer/features/install-nitf-generator/imath_download/Imath_src/src/Imath/ImathBoxAlgo.h"
    "/workspace/.devcontainer/features/install-nitf-generator/imath_download/Imath_src/src/Imath/ImathColor.h"
    "/workspace/.devcontainer/features/install-nitf-generator/imath_download/Imath_src/src/Imath/ImathColorAlgo.h"
    "/workspace/.devcontainer/features/install-nitf-generator/imath_download/Imath_src/src/Imath/ImathEuler.h"
    "/workspace/.devcontainer/features/install-nitf-generator/imath_download/Imath_src/src/Imath/ImathExport.h"
    "/workspace/.devcontainer/features/install-nitf-generator/imath_download/Imath_src/src/Imath/ImathForward.h"
    "/workspace/.devcontainer/features/install-nitf-generator/imath_download/Imath_src/src/Imath/ImathFrame.h"
    "/workspace/.devcontainer/features/install-nitf-generator/imath_download/Imath_src/src/Imath/ImathFrustum.h"
    "/workspace/.devcontainer/features/install-nitf-generator/imath_download/Imath_src/src/Imath/ImathFrustumTest.h"
    "/workspace/.devcontainer/features/install-nitf-generator/imath_download/Imath_src/src/Imath/ImathFun.h"
    "/workspace/.devcontainer/features/install-nitf-generator/imath_download/Imath_src/src/Imath/ImathGL.h"
    "/workspace/.devcontainer/features/install-nitf-generator/imath_download/Imath_src/src/Imath/ImathGLU.h"
    "/workspace/.devcontainer/features/install-nitf-generator/imath_download/Imath_src/src/Imath/ImathInt64.h"
    "/workspace/.devcontainer/features/install-nitf-generator/imath_download/Imath_src/src/Imath/ImathInterval.h"
    "/workspace/.devcontainer/features/install-nitf-generator/imath_download/Imath_src/src/Imath/ImathLine.h"
    "/workspace/.devcontainer/features/install-nitf-generator/imath_download/Imath_src/src/Imath/ImathLineAlgo.h"
    "/workspace/.devcontainer/features/install-nitf-generator/imath_download/Imath_src/src/Imath/ImathMath.h"
    "/workspace/.devcontainer/features/install-nitf-generator/imath_download/Imath_src/src/Imath/ImathMatrix.h"
    "/workspace/.devcontainer/features/install-nitf-generator/imath_download/Imath_src/src/Imath/ImathMatrixAlgo.h"
    "/workspace/.devcontainer/features/install-nitf-generator/imath_download/Imath_src/src/Imath/ImathNamespace.h"
    "/workspace/.devcontainer/features/install-nitf-generator/imath_download/Imath_src/src/Imath/ImathPlane.h"
    "/workspace/.devcontainer/features/install-nitf-generator/imath_download/Imath_src/src/Imath/ImathPlatform.h"
    "/workspace/.devcontainer/features/install-nitf-generator/imath_download/Imath_src/src/Imath/ImathQuat.h"
    "/workspace/.devcontainer/features/install-nitf-generator/imath_download/Imath_src/src/Imath/ImathRandom.h"
    "/workspace/.devcontainer/features/install-nitf-generator/imath_download/Imath_src/src/Imath/ImathRoots.h"
    "/workspace/.devcontainer/features/install-nitf-generator/imath_download/Imath_src/src/Imath/ImathShear.h"
    "/workspace/.devcontainer/features/install-nitf-generator/imath_download/Imath_src/src/Imath/ImathSphere.h"
    "/workspace/.devcontainer/features/install-nitf-generator/imath_download/Imath_src/src/Imath/ImathTypeTraits.h"
    "/workspace/.devcontainer/features/install-nitf-generator/imath_download/Imath_src/src/Imath/ImathVec.h"
    "/workspace/.devcontainer/features/install-nitf-generator/imath_download/Imath_src/src/Imath/ImathVecAlgo.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  execute_process(COMMAND /usr/bin/cmake -E chdir "$ENV{DESTDIR}/usr/local/lib" /usr/bin/cmake -E create_symlink libImath-3_1.so libImath.so)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  message(STATUS "Creating symlink /usr/local/lib/libImath.so -> libImath-3_1.so")
endif()


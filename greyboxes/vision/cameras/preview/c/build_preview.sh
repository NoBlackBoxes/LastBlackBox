#!/bin/bash
set -eu

# Set environment variables
LBBREPO=${LBBROOT}"/repo"
VKREPO="${VKROOT}/repo"

# Set (Cross) Compiler
C_COMPILER="gcc"

# Create output directory
mkdir -p bin
mkdir -p bin/objects
OBJ_FOLDER="bin/objects"

# Set include header directories
INCLUDE_DIRS="\
-I ${VKREPO}/definitions \
-I ${VKREPO}/libraries \
-I ${VKREPO}/drivers \
-I ${VKREPO}/modules"

# Set compile command
COMPILE="${C_COMPILER} -c -Wall -O3 ${INCLUDE_DIRS}"

# Set link command
LINK="${C_COMPILER} -Wall -O3 -lm -lopenblas -lpthread -lgbm -ldrm -lEGL -lGLESv2"

# Compile
${COMPILE} preview.c -o ${OBJ_FOLDER}/preview.o
${COMPILE} ${VKREPO}/libraries/src/parseVK.c -o ${OBJ_FOLDER}/parseVK.o
${COMPILE} ${VKREPO}/libraries/src/mathVK.c -o ${OBJ_FOLDER}/mathVK.o
${COMPILE} ${VKREPO}/libraries/src/visionVK.c -o ${OBJ_FOLDER}/visionVK.o
${COMPILE} ${VKREPO}/drivers/src/vkDisplay_drm.c -o ${OBJ_FOLDER}/vkDisplay.o
${COMPILE} ${VKREPO}/drivers/src/vkCamera_v4l.c -o ${OBJ_FOLDER}/vkCamera.o
${COMPILE} ${VKREPO}/modules/src/vkImage.c -o ${OBJ_FOLDER}/vkImage.o
${COMPILE} -DEGL_NO_X11 -DMESA_EGL_NO_X11_HEADERS ${VKREPO}/modules/src/vkPreview.c -o ${OBJ_FOLDER}/vkPreview.o

# Link
${LINK}\
    ${OBJ_FOLDER}/preview.o \
    ${OBJ_FOLDER}/parseVK.o \
    ${OBJ_FOLDER}/mathVK.o \
    ${OBJ_FOLDER}/visionVK.o \
    ${OBJ_FOLDER}/vkDisplay.o \
    ${OBJ_FOLDER}/vkCamera.o \
    ${OBJ_FOLDER}/vkImage.o \
    ${OBJ_FOLDER}/vkPreview.o \
    -o bin/preview

echo "Done"
exit 0
#FIN

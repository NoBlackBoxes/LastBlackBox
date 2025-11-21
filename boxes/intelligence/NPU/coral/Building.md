# Building Tensorflow Lite Runtime on the Raspberry Pi

## Python Versions
The tflite-runtime on PyPI currently requires **Python 3.11**. This can make it difficult to run the Coral examples on an up-to-date NB3 system. It is possible to install an earlier version of Python.

## Building (on NB3)
- Install required libraries
```bash
sudo apt install libabsl-dev libusb-1.0-0-dev xxd cmake
```
- Install earlier version of flatbuffers
```bash
git clone https://github.com/google/flatbuffers.git
cd flatbuffers/
git checkout v23.5.26
mkdir build && cd build
cmake .. \
    -DFLATBUFFERS_BUILD_SHAREDLIB=ON \
    -DFLATBUFFERS_BUILD_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local
make -j2
sudo make install
cd ../..
```
- Clone tensorflow and checkout desired version: 
```bash
git clone https://github.com/tensorflow/tensorflow
cd tensorflow
git checkout v2.14.0 # (same version of tflite runtime)
cd ..
```
- Clone libedgetpu repo
```bash
git clone https://github.com/feranick/libedgetpu
```
- Update LDFLAGS in Makefile
```make
LIBEDGETPU_LDFLAGS := \
	-Wl,-Map=$(BUILDDIR)/output.map \
	-shared \
	-Wl,--soname,libedgetpu.so.1 \
	-Wl,--version-script=$(BUILDROOT)/tflite/public/libedgetpu.lds \
	-fuse-ld=gold \
	-lflatbuffers \
	-labsl_flags_internal \
	-labsl_flags_marshalling \
	-labsl_str_format_internal \
	-labsl_flags_reflection \
	-labsl_flags_private_handle_accessor \
	-labsl_flags_commandlineflag \
	-labsl_flags_commandlineflag_internal \
	-labsl_flags_config \
	-labsl_flags_program_name \
	-labsl_cord \
	-labsl_cordz_info \
	-labsl_cord_internal \
	-labsl_cordz_functions \
	-labsl_cordz_handle \
	-labsl_hash \
	-labsl_city \
	-labsl_bad_variant_access \
	-labsl_low_level_hash \
	-labsl_raw_hash_set \
	-labsl_bad_optional_access \
	-labsl_hashtablez_sampler \
	-labsl_exponential_biased \
	-labsl_synchronization \
	-labsl_graphcycles_internal \
	-labsl_stacktrace \
	-labsl_symbolize \
	-labsl_debugging_internal \
	-labsl_demangle_internal \
	-labsl_malloc_internal \
	-labsl_time \
	-labsl_civil_time \
	-labsl_time_zone \
	-labsl_strings \
	-labsl_strings_internal \
	-latomic \
	-lrt \
	-labsl_base \
	-labsl_spinlock_wait \
	-labsl_int128 \
	-labsl_throw_delegate \
	-labsl_raw_logging_internal \
	-labsl_log_severity \
	-lusb-1.0
```

- Build
```bash
cd libedgetpu/makefile_build
TFROOT=$HOME/Repos/tensorflow make -j2 libedgetpu
```

- Install
```bash
sudo cp $HOME/Repos/libedgetpu/out/direct/k8/libedgetpu.so.1.0 /usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0
```

project('seticore', ['cpp', 'c'],
        default_options: [
            'buildtype=release',
            'cpp_std=c++14',
            'werror=true',
        ])


cuda_opt = get_option('cuda')
have_cuda = false
cuda_args = []
cpp_args = []

if not cuda_opt.disabled()
    if add_languages('cuda', required: cuda_opt, native: false)
        have_cuda = true
        basic_cuda_args = [ '--std=c++14' ]
        newer_cuda_args = [ '--display-error-number', '--diag-suppress=1675' ]
        
        nvcc_release = run_command('./nvcc_release.sh', check: true).stdout().strip()
        if nvcc_release <= '11.0'
            cuda_args = basic_cuda_args
        else
            cuda_args = basic_cuda_args + newer_cuda_args
        endif

        cpp_args += ['-DSETICORE_CUDA']
    endif
endif

sycl_opt = get_option('sycl')
have_sycl = false

if not sycl_opt.disabled()
    cpp = meson.get_compiler('cpp')
    # Check if compiler is dpcpp/icpx (intel-llvm)
    if cpp.get_id() == 'intel-llvm'
        have_sycl = true
        cpp_args += ['-DSETICORE_SYCL', '-fsycl']
        add_project_link_arguments('-fsycl', language : 'cpp')
    endif
endif

cmake = import('cmake')

fmt_dep = dependency('fmt', required: false)
if not fmt_dep.found()
    fmt_subproj = cmake.subproject('fmt')		
    fmt_dep = fmt_subproj.dependency('fmt')
endif

boost_dep = dependency('boost', modules: [
    'filesystem',
    'program_options',
    'system',
])

hdf5_dep = dependency('hdf5', language: 'c')
fftw3_dep = dependency('fftw3f', required: false)
if fftw3_dep.found()
    cpp_args += ['-DHAVE_FFTW3']
endif

if have_cuda
    cuda_dep = dependency('cuda', version: '>=11', modules: ['cublas', 'cufft'])
else
    cuda_dep = dependency('', required: false)
endif

capnp_dep = dependency('capnp', required: false)
kj_dep = dependency('kj', required: false)

if not capnp_dep.found() or not kj_dep.found()
    capnp_opt = cmake.subproject_options()
    capnp_opt.set_override_option('warning_level', '0')
    capnp_opt.add_cmake_defines({'CMAKE_POSITION_INDEPENDENT_CODE': true, 'BUILD_TESTING': false})
    capnp_subproj = cmake.subproject('capnproto', options: capnp_opt)
    kj_dep = capnp_subproj.dependency('kj')
    capnp_dep = capnp_subproj.dependency('capnp')
endif

deps = [fmt_dep, boost_dep, hdf5_dep, cuda_dep, kj_dep, capnp_dep, fftw3_dep]

common_srcs = [
    'dedoppler.cpp',
    'dedoppler_hit.cpp',
    'dedoppler_hit_group.cpp',
    'dat_file_writer.cpp',
    'event_file_writer.cpp',
    'filterbank_buffer.cpp',
    'filterbank_file_reader.cpp',
    'filterbank_metadata.cpp',
    'fil_reader.cpp',
    'find_events.cpp',
    'h5_reader.cpp',
    'h5_writer.cpp',
    'hit.capnp.c++',
    'hit_file_writer.cpp',
    'hit_recorder.cpp',
    'raw_file.cpp',
    'raw_file_group.cpp',
    'run_dedoppler.cpp',
    'thread_util.cpp',
    'util.cpp',
    'complex_buffer.cpp',
    'device_raw_buffer.cpp',
    'multiantenna_buffer.cpp',
    'multibeam_buffer.cpp',
    'raw_buffer.cpp',
    'upchannelizer.cpp',
    'stamp_extractor.cpp',
    'raw_file_group_reader.cpp',
    'recipe_file.cpp',
    'beamformer.cpp',
]

cuda_srcs = [
    'beamforming_pipeline.cpp', 
    'cuda_util.cu',
    'src/backend/CudaBackend.cu', 
    'taylor.cu',
]

srcs = common_srcs
if have_cuda
    srcs += cuda_srcs
endif

common_tests = [
    'dedoppler_test.cpp',
    'fil_reader_test.cpp',
    'h5_test.cpp',
    'beamformer_test.cpp',
]

cuda_tests = [
    'multibeam_buffer_test.cpp',
    'taylor_test.cu',
]

tests = common_tests
if have_cuda
    tests += cuda_tests
endif


libseticore = static_library('seticore', srcs, dependencies: deps, cuda_args: cuda_args, cpp_args: cpp_args)

# The main binaries

# Only build seticore executable if we have CUDA or if main.cpp is refactored (it is not yet)
# But main.cpp uses Dedopplerer, which is refactored.
# It depends on beamformer though? 
# Let's try to build it and see if it links.
executable('seticore', ['main.cpp'],
           dependencies: deps,
           link_with: libseticore,
           cpp_args: cpp_args)

executable('tests', tests + ['tests.cpp'],
           dependencies: deps,
           link_with: libseticore,
           cuda_args: cuda_args,
           cpp_args: cpp_args)

# Other binaries need check for dependencies
if have_cuda
    executable('beamforming_integration_test',
               ['beamforming_integration_test.cpp'],
               dependencies: deps,
               link_with: libseticore,
               cpp_args: cpp_args)
endif

executable('blockls',
           ['blockls.cpp'],
           dependencies: deps,
           link_with: libseticore,
           cpp_args: cpp_args)

executable('hitls',
           ['hitls.cpp'],
           dependencies: deps,
           link_with: libseticore,
           cpp_args: cpp_args)

if have_cuda
    executable('file_io_benchmark',
               ['file_io_benchmark.cpp'],
               dependencies: deps,
               link_with: libseticore,
               cpp_args: cpp_args)
endif

executable('extract',
           ['extract.cpp'],
           dependencies: deps,
           link_with: libseticore,
           cpp_args: cpp_args)

executable('mmap_benchmark',
           ['mmap_benchmark.cpp'],
           dependencies: deps,
           link_with: libseticore,
           cpp_args: cpp_args)

executable('rawls',
           ['rawls.cpp'],
           dependencies: deps,
           link_with: libseticore,
           cpp_args: cpp_args)

if have_cuda
    executable('taylor_benchmark',
               ['taylor_benchmark.cu'],
               dependencies: deps,
               link_with: libseticore,
               cuda_args: cuda_args,
               cpp_args: cpp_args)
endif

executable('recipels',
           ['recipels.cpp'],
           dependencies: deps,
           link_with: libseticore,
           cpp_args: cpp_args)

executable('stampls',
           ['stampls.cpp'],
           dependencies: deps,
           link_with: libseticore,
           cpp_args: cpp_args)

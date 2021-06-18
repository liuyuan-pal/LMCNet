from __future__ import division, absolute_import, with_statement, print_function
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob
import os

_ext_src_root = os.path.abspath(".")
# _ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob(
#     "{}/src/*.cu".format(_ext_src_root)
# )
# _ext_headers = glob.glob("{}/include/*".format(_ext_src_root))

_ext_headers = ['./include/ball_query.h','./include/cuda_utils.h','./include/group_points.h']
_ext_sources = ['./src/ball_query.cpp','./src/ball_query_gpu.cu','./src/bindings.cpp','./src/group_points.cpp','./src/group_points_gpu.cu']

requirements = ["etw_pytorch_utils==1.1.1", "h5py", "pprint", "enum34", "future"]

setup(
    name="pointnet2_exts",
    author="Erik Wijmans",
    # packages=find_packages(),
    # install_requires=requirements,
    ext_modules=[
        CUDAExtension(
            "pointnet2_exts",
            _ext_sources,
            extra_compile_args={
                "cxx": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
                "nvcc": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
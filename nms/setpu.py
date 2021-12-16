from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        'cpu_nms',  # 生成的动态链接库的名字
        sources=['cpu_nms.cpu_nms_pyx'],
        language='c',
        include_dirs=[numpy.get_include()],  # 传给 gcc 的 -I 参数
        library_dirs=[],  # 传给 gcc 的 -L 参数
        libraries=[],  # 传给 gcc 的 -l 参数
        extra_compile_args=[],  # 传给 gcc 的额外的编译参数
        extra_link_args=[]  # 传给 gcc 的额外的链接参数
    ),
    Extension(
        'cpu_soft_nms',  # 生成的动态链接库的名字
        sources=['cpu_soft_nms.cpu_nms_pyx'],
        language='c',
        include_dirs=[numpy.get_include()],  # 传给 gcc 的 -I 参数
        library_dirs=[],  # 传给 gcc 的 -L 参数
        libraries=[],  # 传给 gcc 的 -l 参数
        extra_compile_args=[],  # 传给 gcc 的额外的编译参数
        extra_link_args=[]  # 传给 gcc 的额外的链接参数
    )
]
setup(name='nms',
      ext_modules=cythonize(extensions))

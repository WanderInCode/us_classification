from setuptools import setup, find_packages


setup(
    name="ResCls",
    version="1.0",
    author="zhanyh",
    author_email="zhanyh123@foxmail.com",
    description=(""),
    license="MIT",
    keywords="",
    url="",
    packages=find_packages(),  # 需要打包的目录列表

    # 需要安装的依赖
    install_requires=[
        'keras>=2.1.5',
        'tenforflow>=1.6.0',
        'numpy>=1.14.2',
        'cv2>=3.4.0'
    ],

    zip_safe=False
)

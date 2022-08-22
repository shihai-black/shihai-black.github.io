## PaddleOCR踩坑日记

## 背景

由于项目需要，调研了一下paddleocr在文字识别的效果，在离线服务器部署完成，但是迁移线上时出现了极为麻烦的环境配置问题，所以特意写下这篇文章，以防日后出现相同的情况。

## 具体安装步骤

- 检查系统版本[我是16.04版本]

```sh
lsb_release -a

No LSB modules are available.
Distributor ID:	Ubuntu
Description:	Ubuntu 16.04.7 LTS
Release:	16.04
Codename:	xenial
```

- 根据系统版本安装对应的cuda版本【我是11.2版本，主要是Paddle-gpu适配的是10.2/11.2/11.6】
  - 不用单独安装nvidia驱动，安装cudatoolkit时会自动匹配上，如果有Nvidia驱动直接nvidia-uninstall卸载即可
  - 查看paddleocr允许的cuda版本

- 根据对应的cuda版本安装pytorch【版本比使用低一些不要紧】

```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu111
```

- 根据对应的cuda安装paddleocr【我装的是2.2.2】
  - **要求！！！CUDA 11.2，cuDNN 8.1.1（多卡环境下 NCCL>=2.7）**

```sh
python -m pip install paddlepaddle-gpu==2.2.2.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

- 根据cuda，paddleocr安装cudnn8.1.1
  - 首先下载压缩包

  - ```sh
    tar -xvf cudnn-11.2-linux-x64-v8.1.1.33.tgz
    cd cuda
    cp lib64/* /usr/local/cuda-11.2/lib64/
    cp include/* /usr/local/cuda-11.2/include/
    chmod a+r /usr/local/cuda-11.2/lib64/* /usr/local/cuda-11.2/include/*
    ```

  - 把下面这段话加到～/.bashrc文件内，然后source ～/.bashrc

    ```sh
    # add nvcc compiler to path
    export PATH=$PATH:/usr/local/cuda-11.2/bin
    # add cuBLAS, cuSPARSE, cuRAND, cuSOLVER, cuFFT to path
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.2/lib64:/usr/lib/x86_64-linux-gnu
    ```

- 安装paddleocr

  - ```sh
    paddleocr --image_dir "https://marketing-publication.s3.us-west-2.amazonaws.com/tmp/ocr_bill/images/bill/1.png" --use_angle_cls true --lang en --use_gpu True 
    ```

## 链接

- [cuda安装教程](https://zhuanlan.zhihu.com/p/520536351)
- [cuda网站](https://developer.nvidia.com/cuda-11.2.0-download-archive)
- [pytorch下载地址](https://pytorch.org/)
- [paddle安装地址](https://www.paddlepaddle.org.cn/install/old?docurl=/documentation/docs/zh/install/pip/linux-pip.html)
- [cudnn地址](https://developer.nvidia.com/rdp/cudnn-archive)
- [cudnn教程](https://blog.csdn.net/qq_44961869/article/details/115954258)


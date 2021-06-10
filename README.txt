项目参考了滴滴开源的athena-signal和breizhn开源的DTLN,并参考athena中ns模块实现了dtln模块，
dtln模块是基于DTLN中的tflite预训练模型进行推理实现频率和时域的降噪处理的。

个人不是很擅长写Makefile，用简单的脚本实现编译工作。希望有能力的朋友帮忙改进成Makefile或CMake。

用法比较简单：

./compile_lib.sh
在lib目录下生成libathena.so

./compile_exm.sh
在bin目录下生成dtln

bin/dtln data/airconditioner.wav data/airconditioner_dtln_ns_agc.wav data/airconditioner_vad.wav

在examples/ns.c中如果设置了NS_KEY/DTLN_KEY/AGC_KEY，会按照会 dtln->ns->vad->agc 的流程执行

athena-signal中的ns对一些随机噪声还是比较有效的，因此放在DTLN之后，

如果放在DTLN之前会引入一些不理想的效果，猜测是传统ns之后影响了DTLN的推理

备注：
(1)编译libathena.so依赖tflite，请参考如下官方链接下载编译
https://tensorflow.google.cn/lite/guide/build_arm64
其中x86 linux按照替换掉链接中的：
./tensorflow/lite/tools/make/build_aarch64_lib.sh
为：
./tensorflow/lite/tools/make/build_lib.sh

(2)编译dtlb依赖libsndfile读写wav文件，可自行编译

(3)thirdpart目录下面已经包含了，在centos7上编译得到的静态库

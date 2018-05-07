# TF Mobile and TF lite
目前 TF Lite 还没有成熟，代码中散落各类实现和例子，有必要先了解一下两个相近的框架 TF Mobile 和 TF Lite。TF Mobile 早于 TF Lite，是Tensorflow最初在移动设备上的移植，它没有硬件加速，没有量化等优化，因此可以认为它是一个过度框架。而 TF Lite 实现了比 protobuf 更轻量级的 flatbuffer，开销小，但目前不是所有的 op 都支持。总之，能用 TF Lite 的地方就尽量用，不能用再考虑 TF Mobile。

## TF Mobile demo analysis
TF Mobile Camera 例子涵盖最基础的四个应用场景，classification, detection, stylize 和 sppech。

org.tensorflow.demo java库，主要API： TensorFlowInferenceInterface

如何使用这个类可以参考[TensorFlowImageClassifier.java](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/android/src/org/tensorflow/demo/TensorFlowImageClassifier.java)

java 程序如果想用 tensorflow，可以使用 jcenter 仓库中的版本，在 build.gradle 加上如下几行即可：
```
allprojects {
    repositories {
        jcenter()
    }
}

dependencies {
    compile 'org.tensorflow:tensorflow-android:+'
}
```
java apk依赖两个c++ 库：

* libtensorflow_demo.so
* libtensorflow_inference.so

其中 libtensorflow_demo.so 做 RGB->YUV转换，并不重要。而 libtensorflow_inference.so 是 tensorflow C++库，有必要搞明白它怎么来的，便于以后自己编译。

```
$ cd $TF_REPO_ROOT
$ bazel build -c opt //tensorflow/contrib/android:libtensorflow_inference.so \
     --crosstool_top=//external:android/crosstool \
     --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
     --cpu=armeabi-v7a
```
把cpu替换成自己需要的架构。更详细信息参考[这里](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/android)

要看demo效果，可以直接使用 nightly build 的demo，也可以用 android studio 打开 tensorflow/examples/android 目录进行编译。参考[这里](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android)

## TF Lite demo


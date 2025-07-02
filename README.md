# ailia SDK JNI Package

!! CAUTION !! “ailia” IS NOT OPEN SOURCE SOFTWARE (OSS). As long as user complies with the conditions stated in License Document, user may use the Software for free of charge, but the Software is basically paid software.

## About this package

Demo project of ailia SDK with Android Studio (Kotlin)

## Test environment

- macOS 12.1 / Windows 11
- Android Studio 2023.1.1
- Gradle 7.2
- ailia SDK 1.5.0

## Detail

This demo estimate keypoints of person.

Input

<img src="./app/src/main/res/raw/person.jpg" width=480 height=480/>

Output

![input image](./demo.png)

## Main code

[MainActivity.kt](/app/src/main/java/jp/axinc/ailia_kotlin/MainActivity.kt)

## About ailia SDK

ailia SDK is a cross-platform high speed inference SDK. The ailia SDK provides a consistent C++ API on Windows, Mac, Linux, iOS, Android, Jetson and RaspberryPi. It supports Unity, Python and JNI for efficient AI implementation. The ailia SDK makes great use of the GPU via Vulkan and Metal to serve accelerated computing.

# ailia-android-studio-kotlin

Demo project of ailia SDK with Android Studio (Kotlin)

## Test environment

- macOS 12.1 / Windows 11
- Android Studio 2023.1.1
- Gradle 7.2
- ailia SDK 1.5.0

## Setup

Please put ailia libraries here.

```
app/src/main/jniLibs/arm64-v8a/libailia.so
app/src/main/jniLibs/armeabi-v7a/libailia.so
app/src/main/jniLibs/x86/libailia.so
app/src/main/jniLibs/x86_64/libailia.so
```

## Detail

This demo estimate keypoints of person.

Input

<img src="./app/src/main/res/raw/person.jpg" width=480 height=480/>

Output

![input image](./demo.png)

## Main code

[MainActivity.kt](/app/src/main/java/jp/axinc/ailia_kotlin/MainActivity.kt)
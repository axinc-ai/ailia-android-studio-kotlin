# ailia-kotlin

Demo project of ailia SDK with Android Studio (Kotlin)

## Test environment

- macOS 12.1
- Android Studio 2020.3.1
- ailia SDK 1.2.9

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

![input image](./app/src/main/res/raw/person.jpg)

Output (Console)

```
I/AILIA_Main: objCount (human count) = 1
I/AILIA_Main: total score = 25.370672
I/AILIA_Main: angle[3] = {-0.11134362, 0.0, 0.0}
    keypoint[0] = {x: 0.753125, y: 0.51666665, z_local: 0.0, score: 0.8854157, interpolated: 0}
    keypoint[1] = {x: 0.76875, y: 0.49166667, z_local: 0.0, score: 0.8802486, interpolated: 0}
    keypoint[2] = {x: 0.740625, y: 0.5083333, z_local: 0.0, score: 0.9541165, interpolated: 0}
    keypoint[3] = {x: 0.809375, y: 0.49583334, z_local: 0.0, score: 0.7805725, interpolated: 0}
I/AILIA_Main: keypoint[4] = {x: 0.734375, y: 0.51666665, z_local: 0.0, score: 0.15920368, interpolated: 0}
    keypoint[5] = {x: 0.859375, y: 0.6166667, z_local: 0.0, score: 0.81659067, interpolated: 0}
    keypoint[6] = {x: 0.73125, y: 0.6166667, z_local: 0.0, score: 0.7995946, interpolated: 0}
    keypoint[7] = {x: 0.884375, y: 0.7583333, z_local: 0.0, score: 0.8942372, interpolated: 0}
    keypoint[8] = {x: 0.665625, y: 0.71666664, z_local: 0.0, score: 0.93179995, interpolated: 0}
    keypoint[9] = {x: 0.921875, y: 0.8958333, z_local: 0.0, score: 0.84773785, interpolated: 0}
    keypoint[10] = {x: 0.696875, y: 0.59583336, z_local: 0.0, score: 0.8966119, interpolated: 0}
I/AILIA_Main: keypoint[11] = {x: 0.8, y: 0.87083334, z_local: 0.0, score: 0.6003204, interpolated: 0}
    keypoint[12] = {x: 0.715625, y: 0.85833335, z_local: 0.0, score: 0.7168967, interpolated: 0}
    keypoint[13] = {x: 0.815625, y: 0.99583334, z_local: 0.0, score: 0.118452325, interpolated: 0}
    keypoint[14] = {x: 0.684375, y: 0.99583334, z_local: 0.0, score: 0.22999896, interpolated: 0}
    keypoint[15] = {x: 0.0, y: 0.0, z_local: 0.0, score: 0.0, interpolated: 0}
    keypoint[16] = {x: 0.0, y: 0.0, z_local: 0.0, score: 0.0, interpolated: 0}
    keypoint[17] = {x: 0.79375, y: 0.6166667, z_local: 0.0, score: 0.9140342, interpolated: 0}
    keypoint[18] = {x: 0.76979166, y: 0.78194445, z_local: 0.0, score: 0.6003204, interpolated: 1}
I/AILIA_Main: Success
```

## Main code

[MainActivity.kt](/app/src/main/java/jp/axinc/ailia_kotlin/MainActivity.kt)
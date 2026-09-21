# MobileNetV2-Quantization
Exercise: Quantizing ML vision models for processing on the edge

We take an existing baseline model or train one from scratch, and apply quantization techniques. We work in *PyTorch* and apply:

#### (1) Dynamic PTQ,

#### (2) Static PTQ via FX Graph Mode Quantization.

Evaluation is interpreted by comparing how the model's architecture responds to the different methods. 

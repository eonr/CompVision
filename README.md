# CNN-Tests
## MNIST-1
<b>Layer-1</b>: Conv2d (5x5 kernels, 32 filters) + Max pooling (2, 2)  <br>
<b>Layer-2</b>: Conv2d (5x5,64 filters) + Max pooling (2, 2) + dropout<br>
<b>Layer-3</b>: Linear (1024 inputs, 10 outputs) + RelU <br>
<br>
## MNIST-2
<b>Layer-1</b>: Conv2d (5x5 kernels, 32 filters) + Max pooling (2, 2) <br>
<b>Layer-2</b>: Conv2d (5x5,64 filters) + Max pooling (2, 2) + dropout <br>
<b>Layer-3</b>: Conv2d (3x3 kernels, 64 filters, padding = 1) <br>
<b>Layer-4</b>: Conv2d (3x3 kernels, 64 filters, padding = 1) + Output of Layer-2 (Residual) <br>
<b>Layer-5</b>: Linear (1024 inputs, 10 outputs) + RelU <br>

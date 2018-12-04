# EveNet: Expression of Emotion and Visimes Network

[![Build Status](https://travis-ci.org/elggem/EveNet.svg?branch=master)](https://travis-ci.org/elggem/EveNet)

This is a fork of ibab's excellent implementation of WaveNet. Here we are implementing changes for the generation of facial animations.
## Exprements
### Network with softmax output layer
Authors of wavenet stated that modeling the conditional distribution:    

![eq1](http://www.sciweavers.org/tex2img.php?eq=p%28%20x_%7Bt%7D%7C%20x_%7B1%7D%2Cx_%7B2%7D%2C...%2Cx_%7Bt-1%7D%29%20%20&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0%22%20align=%22center%22%20border=%220%22%20alt=%22p(%20x_{t}|%20x_{1},x_{2},...,x_{t-1}))  

with softmax distribution tends to work better than some of previous modeling, such as mixture density network or mixture of conditional Gaussian scale mixtures. 

This approach models each shape key values of each frame to softmax probability distribution of 11 classes. The probability values are calculated with the method explained below.

More on the algorithm used to convert real value to softmax ditribution is found [here](https://docs.google.com/document/d/1PTGRjHrIJsW_7Ypv6uc3etWj37sevNbXCcDpuH8S1b8/edit?ts=5bfd4948#bookmark=id.43gy6l9nzf80).

Here is [github branch](https://github.com/hansonrobotics/Evenet/tree/expermenting-with-softmax-layer) for this experment.


### Shape key sampling
This approach is variant of approach adopted in pixelcnn, to extend the dependencies among pixels to color channels. In pixelcnn paper the author stated the joint probability for pixel sampling is  

![equation](http://www.sciweavers.org/tex2img.php?eq=p%28x%29%20%3D%20%20%5Cprod_%7Bi%3D1%7D%5E%7Bn%5E2%7Dp%28x_i%7Cx_%7B%3Ci%7D%29%20&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0),  

Where, ![xi](http://www.sciweavers.org/tex2img.php?eq=x_i&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0) is ![ith](http://www.sciweavers.org/tex2img.php?eq=i%5E%7Bth%7D%20&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0) pixel.  
They also extend this pixel dependencies to color channels as follows:-

![img](http://www.sciweavers.org/tex2img.php?eq=%0Ap%28%20x_%7Bi%7D%7Cx_%7B%3Ci%7D%29%20%3D%20p%28x_%7Bi%2CR%7D%7Cx_%7B%3Ci%7D%29%20%2A%20p%28x_%7Bi%2CG%7D%7Cx_%7B%3Ci%7D%2C%20x_%7Bi%2C%20R%7D%29%20%2A%20p%28x_%7Bi%2CB%7D%7Cx_%7B%3Ci%7D%2C%20x_%7Bi%2C%20R%7D%2C%20x_%7Bi%2C%20G%7D%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=00)

If we use indexes for color channels(assign 1 to R index, 2 to G index and 3 to B index), the joint probability p(x) now can be written as   
![eq4](http://www.sciweavers.org/tex2img.php?eq=p%28x%29%20%3D%20%20%5Cprod_%7Bi%3D1%7D%5E%7Bn%5E2%7D%20%20%5Cprod_%7Bj%3D1%7D%5E3%20p%28x_%7Bi%2Cj%7D%7Cx_%7B%3Ci%7D%2Cx_%7Bi%2C%3Cj%7D%29%20&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)  

This type of joint probability can be applied to shape key sampling. Now the above joint probability can be rewritten for shape key sampling as  
![eq6](http://www.sciweavers.org/tex2img.php?eq=p%28x%29%20%3D%20%20%5Cprod_%7Bi%3D1%7D%5ES%20%20%5Cprod_%7Bj%3D1%7D%5EN%20%20p%28x_%7Bi%2Cj%7D%7Cx_%7B%3Ci%7D%2Cx_%7Bi%2C%3Cj%7D%29%20&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)  
where S is number of samples, N is number of shape keys in single frame, ![xi](http://www.sciweavers.org/tex2img.php?eq=x_i&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0) is  ![ith](http://www.sciweavers.org/tex2img.php?eq=i%5E%7Bth%7D%20&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0) frame and ![xi,j](http://www.sciweavers.org/tex2img.php?eq=x_%7Bi%2Cj%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0) is jth shape key of ith frame.


Code implementation  for this experment is found on [github branch](https://github.com/hansonrobotics/Evenet/tree/shape-key-sampling-with-softmax-layer).

## Requirements

TensorFlow needs to be installed before running the training script.
Code is tested on TensorFlow version 1.0.1 for Python 2.7 and Python 3.5.

In addition, [librosa](https://github.com/librosa/librosa) must be installed for reading and writing audio.

To install the required python packages, run
```bash
pip install -r requirements.txt
```

For GPU support, use
```bash
pip install -r requirements_gpu.txt
```

## Running tests

Install the test requirements
```
pip install -r requirements_test.txt
```

Run the test suite
```
./ci/test.sh
```

## Related projects

- [tensorflow-wavenet](https://github.com/ibab/tensorflow-wavenet), the WaveNet implementation this is based on.
![f1]: http://chart.apis.google.com/chart?cht=tx&chl=p(x|%20x_1,%20x_2,%2E%2E%2E,x_{t-1})
![f2]: http://chart.apis.google.com/chart?cht=tx&chl=p(x)%20=\prod_{i=1}^{n^2}p(x_i|x{%3Ci})
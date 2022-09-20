sudo apt-get update
sudo apt install libopenblas-dev liblapack3 build-essential cmake gfortran python3.7 libhdf5-serial-dev protobuf-compiler

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1Wl8b36UJWPNlkT4v4gQiKkSkD71SN64K" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1Wl8b36UJWPNlkT4v4gQiKkSkD71SN64K" -o bazel
sudo chmod +x bazel
sudo cp ./bazel /usr/local/bin
rm ./bazel

git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout v1.14.0
./configure
bazel build //tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
pip install /tmp/tensorflow_pkg/tensorflow-VERSION-TAGS.whl

wget https://github.com/rockchip-linux/rknn-toolkit/releases/download/v1.7.3/rknn-toolkit-v1.7.3-packages.zip
unzip rknn-toolkit-v1.7.3-packages.zip -d rknn_toolkit
cd rknn_toolkit
ln -s /usr/bin/python3.7 /usr/bin/python
python -m pip install rknn_toolkit-1.7.3-cp37-cp37m-linux_aarch64.whl
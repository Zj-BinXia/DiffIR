sudo apt-get update
sudo apt install tmux
sudo apt install libgl1-mesa-glx
pip install basicsr --index-url http://pypi.douban.com/simple --trusted-host pypi.douban.com
pip install -r requirements.txt --index-url http://pypi.douban.com/simple --trusted-host pypi.douban.com
pip install pandas --index-url http://pypi.douban.com/simple --trusted-host pypi.douban.com
sudo python3 setup.py develop
pip install einops --index-url http://pypi.douban.com/simple --trusted-host pypi.douban.com
pip install natsort --index-url http://pypi.douban.com/simple --trusted-host pypi.douban.com
pip uninstall nvidia_cublas_cu11

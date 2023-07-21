sudo apt-get update
sudo apt install tmux
sudo apt install libgl1-mesa-glx
pip3 install basicsr --index-url http://pypi.douban.com/simple --trusted-host pypi.douban.com
pip3 install -r requirements.txt --index-url http://pypi.douban.com/simple --trusted-host pypi.douban.com
pip3 install pandas --index-url http://pypi.douban.com/simple --trusted-host pypi.douban.com
sudo python3 setup.py develop
#sudo pip uninstall bytedmetrics
pip3 install einops --index-url http://pypi.douban.com/simple --trusted-host pypi.douban.com
pip3 install lpips --index-url http://pypi.douban.com/simple --trusted-host pypi.douban.com

pip3 install torchsummary --index-url http://pypi.douban.com/simple --trusted-host pypi.douban.com
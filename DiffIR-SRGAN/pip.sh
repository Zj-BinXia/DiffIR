sudo apt-get update
sudo apt install tmux
sudo apt install libgl1-mesa-glx
pip install basicsr --index-url http://pypi.douban.com/simple --trusted-host pypi.douban.com
pip install -r requirements.txt --index-url http://pypi.douban.com/simple --trusted-host pypi.douban.com
pip install pandas --index-url http://pypi.douban.com/simple --trusted-host pypi.douban.com
sudo python3 setup.py develop
#sudo pip uninstall bytedmetrics
pip install einops --index-url http://pypi.douban.com/simple --trusted-host pypi.douban.com
pip install lpips --index-url http://pypi.douban.com/simple --trusted-host pypi.douban.com

pip install torchsummary --index-url http://pypi.douban.com/simple --trusted-host pypi.douban.com

pip install timm --index-url http://pypi.douban.com/simple --trusted-host pypi.douban.com

pip install thop --index-url http://pypi.douban.com/simple --trusted-host pypi.douban.com

pip install ptflops --index-url http://pypi.douban.com/simple --trusted-host pypi.douban.com
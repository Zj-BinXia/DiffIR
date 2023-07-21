sudo apt-get update
sudo apt install tmux
sudo apt install libgl1-mesa-glx
pip install -r requirements.txt --index-url http://pypi.douban.com/simple --trusted-host pypi.douban.com
sudo apt-get install zip

pip install detectron2 --index-url http://pypi.douban.com/simple --trusted-host pypi.douban.com
python3 -m pip3 install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

pip install einops --index-url http://pypi.douban.com/simple --trusted-host pypi.douban.com
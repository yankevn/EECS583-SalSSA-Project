if [ -f /etc/lsb-release ]; then 
  yes | sudo apt install gcc g++ binutils
  yes | sudo apt install gcc-multilib g++-multilib
  yes | sudo apt install cmake
  yes | sudo apt install python python-pip python-tk
  sudo pip install --upgrade pip
  sudo pip install numpy
  sudo pip install matplotlib
  sudo pip install seaborn
elif [ -f /etc/SuSE-release ]; then 
  yes | sudo apt-get install gcc gcc-c++ binutils
  yes | sudo apt-get install gcc-32bit gcc-c++-32bit
  yes | sudo apt-get install cmake
  yes | sudo apt-get install python python-pip python-tk
  sudo pip install --upgrade pip
  sudo pip install numpy
  sudo pip install matplotlib
  sudo pip install seaborn
else
  yes | sudo apt-get install gcc gcc-c++ binutils
  yes | sudo apt-get install gcc-32bit gcc-c++-32bit
  yes | sudo apt-get install cmake
  yes | sudo apt-get install python python-pip python-tk
  sudo pip install --upgrade pip
  sudo pip install numpy
  sudo pip install matplotlib
  sudo pip install seaborn
fi


export DEEPLEARNING=$PWD
export PYTHONPATH=$DEEPLEARNING/..:$PYTHONPATH
export PATH=$DEEPLEARNING:$PATH
# Install all dependencies listed in the requirements.txt file
pip3 install -r requirements.txt
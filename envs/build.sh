ROOT=${PWD} 
CONDA_PATH=$(conda info --base)

source ${CONDA_PATH}/etc/profile.d/conda.sh
conda create -y -n active-gs python=3.9 cmake=3.14.0
conda activate active-gs

export PYTHONNOUSERSITE=True

# install habitat simulator 0.2.4
cd ${ROOT}/simulator
git clone git@github.com:liren-jin/habitat-sim.git
cd habitat-sim
pip install -r requirements.txt
python setup.py install --headless --bullet

# install acive-gs package support
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r ${ROOT}/envs/requirements.txt
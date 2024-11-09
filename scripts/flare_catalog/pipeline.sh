#!/bin/bash

git clone https://github.com/DEVANSH-DVJ/SuryaDrishti.git

if [ -z "$xsmdas" ]
then
    echo "xsmdas is not installed"
    echo "Installing xsmdas"
    wget https://www.prl.res.in/ch2xsm/static/ch2_xsmdas_20210628_v1.2.zip
    unzip ch2_xsmdas_20210628_v1.2.zip
    rm ch2_xsmdas_20210628_v1.2.zip

    export xsmdas=$(pwd)/xsmdas
    export PATH="$PATH:$xsmdas/bin:$xsmdas/scripts"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$xsmdas/lib"
    export PFILES="$xsmdas/pfiles"

    echo "export xsmdas=$(pwd)/xsmdas" >> ~/.bashrc
    echo "export PATH=\"\$PATH:\$xsmdas/bin:\$xsmdas/scripts\"" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=\"\$LD_LIBRARY_PATH:\$xsmdas/lib\"" >> ~/.bashrc
    echo "export PFILES=\"\$xsmdas/pfiles\"" >> ~/.bashrc

    cd $xsmdas
    ./InstallLibs

    wget https://www.prl.res.in/ch2xsm/static/ch2_xsm_caldb_20210628.zip  

    unzip ch2_xsm_caldb_20210628.zip -d $xsmdas
    rm ch2_xsm_caldb_20210628.zip

    cd $xsmdas
    make
    cd -
fi

cd SuryaDrishti

source ./scripts/extract_lc.sh
source ./scripts/extract_pha.sh
source ./scripts/xsm_genspec_batch.sh
python3 ./scripts/xsm_gen_lc.py
source ./scripts/list_data.sh
mkdir ./data/Lightcurves
mv ./data/XSM_Generated_LightCurve/* ./data/Lightcureves

python3 ./scripts/xsm_summary.py
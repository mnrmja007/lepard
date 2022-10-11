#!/bin/bash

echo "Welcome to Lepard!!!"
echo "Do you want to test on default data? (yes/no)"
read default

yes=yes
if [[ "$default" == "$yes" ]]; then
    echo "Please enter the registration mode - (rigid / non-rigid): "
    read reg_mode
    rigid=rigid
    if [[ "$reg_mode" == "$rigid" ]]; then
        echo "Do you want to run on low match data? (yes/no)"
        read low_match
        if [[ "$low_match" == "$yes" ]]; then
            python main.py configs/test/3dmatch_default_low_match.yaml
        else
            python main.py configs/test/3dmatch_default.yaml
        fi
    else
        echo "Do you want to run on low match data? (yes/no)"
        read low_match
        if [[ "$low_match" == "$yes" ]]; then
            python main.py configs/test/4dmatch_default_low_match.yaml
        else
            python main.py configs/test/4dmatch_default.yaml
        fi
    fi

else

    # Ask the user for path to base directory
    echo "Please enter the path to the experiment directory: "
    read expt_dir
    echo "Experiment directory: $expt_dir"

    # Read the source and target point clouds
    source_point_cloud_path=$expt_dir/src.ply
    target_point_cloud_path=$expt_dir/tgt.ply

    # Read the mode of registration
    echo "Please enter the registration mode - (rigid / non-rigid): "
    read reg_mode

    # Preprocess the source and target point clouds and save them in the required format
    python generate_data_for_lepard.py -m $reg_mode -s $source_point_cloud_path -t $target_point_cloud_path -o $expt_dir

    # Copy the processed files to the required directory
    lepard_base=/home/gridraster/masters/lepard
    echo "Lepard base path: $lepard_base"
    # echo "Please enter the base path of Lepard: "
    # read lepard_base

    data_dir_for_lepard=$lepard_base
    rigid=rigid
    if [[ "$reg_mode" == "$rigid" ]]; then
        data_dir_for_lepard=$lepard_base/data/3dmatch/custom_test
        cp $expt_dir/source_point_cloud.pth $data_dir_for_lepard
        cp $expt_dir/target_point_cloud.pth $data_dir_for_lepard
        cp $expt_dir/processed_inputs.pkl $data_dir_for_lepard
    else
        data_dir_for_lepard=$lepard_base/data/split/4DMatch_custom_test/custom
        cp $expt_dir/processed_inputs.npz $data_dir_for_lepard
    fi
    echo "Data has been successfully written to: $data_dir_for_lepard. We are ready to run Lepard."

    if [[ "$reg_mode" == "$rigid" ]]; then
        python main.py configs/test/3dmatch_custom.yaml
        cd $lepard_base/results/3dmatch/
        cp -r `ls -td -- */ | head -n 1 | cut -d'/' -f1` $expt_dir
    else
        python main.py configs/test/4dmatch_custom.yaml
        cd $lepard_base/results/4dmatch/
        cp -r `ls -td -- */ | head -n 1 | cut -d'/' -f1` $expt_dir
    fi
fi
import os
import shutil
import zipfile
import tempfile
import argparse
import nibabel as nib
import numpy as np
import glob
import random
import matplotlib.pyplot as plt


# arguments part
parser = argparse.ArgumentParser()
parser.add_argument('--zip_path', type=str, default=None, help='UNC paired 3T-7T dataset zip file path')
parser.add_argument('--dataset_name', type=str, default='unc', help='dataset name')
parser.add_argument('--dataset_root', type=str, default='./dataset', help='root folder for preprocessed data')


# data preprocessing function for UNC datasets
def unc_nii_to_npy(nii_dir, npy_dir):
    # only use 3T data, which is ses-1, now get all available
    # domain A - T1w, domain B - T2w
    # create directories for domain A B train and test
    trainA_dir = os.path.join(npy_dir, 'trainA')
    os.makedirs(trainA_dir, exist_ok=True)
    
    trainB_dir = os.path.join(npy_dir, 'trainB')
    os.makedirs(trainB_dir, exist_ok=True)
    
    testA_dir = os.path.join(npy_dir, 'testA')
    os.makedirs(testA_dir, exist_ok=True)
    
    testB_dir = os.path.join(npy_dir, 'testB')
    os.makedirs(testB_dir, exist_ok=True)
    
    # get NIFTI files list using T1w part
    flist = glob.glob(os.path.join(nii_dir, 'dataset/Aligned/*/ses-1/anat/*_T1w_*.nii.gz'))
    for t1w_nii_path in flist:
        sub_id = t1w_nii_path.split('/')[-1].split('_')[0]
        print(f'Now preprocessing data of {sub_id}')
        t2w_nii_path = t1w_nii_path.replace('_T1w_', '_T2w_')
        if not os.path.isfile(t2w_nii_path):
            raise ValueError(f'{t2w_nii_path} does not exist')
        # load the NIFTI files
        t1_nib = nib.load(t1w_nii_path)
        t2_nib = nib.load(t2w_nii_path)
        # get the data arrays
        t1_data = t1_nib.get_fdata()
        t2_data = t2_nib.get_fdata()
        # use all data
        #for idx in range(t1_data.shape[-1]):
        # skip incomplete data
        for idx in range(120, 280):
            t1_slice = t1_data[:, :, idx]
            # skip the slice if t1w data all zero
            if np.abs(t1_slice.min() - t1_slice.max()) < 1e-6:
                continue
            # normalize t1w data
            t1_slice = (t1_slice - t1_slice.min()) / (t1_slice.max() - t1_slice.min())
            t2_slice = t2_data[:, :, idx]
            # skip the slice if t2w data all zero
            if np.abs(t2_slice.min() - t2_slice.max()) < 1e-6:
                continue
            # normalize t2w data
            t2_slice = (t2_slice - t2_slice.min()) / (t2_slice.max() - t2_slice.min())
            fname = sub_id + '_' + str(idx) + '.npy'
            # randomly select two subjects data for evaluation
            if sub_id in ['sub-02', 'sub-09']:
                np.save(os.path.join(testA_dir, fname), t1_slice)
                np.save(os.path.join(testB_dir, fname), t2_slice)
            else:
                np.save(os.path.join(trainA_dir, fname), t1_slice)
                np.save(os.path.join(trainB_dir, fname), t2_slice)
    # generate samples image
    testA_npy_list = glob.glob(os.path.join(testA_dir, '*.npy'))
    random.shuffle(testA_npy_list)
    # a figure with 4 rows and 2 columns
    fig, axs = plt.subplots(4, 2, figsize=(8, 16))
    for i in range(4):
        print(testA_npy_list[i])
        t1_npy = np.load(testA_npy_list[i])
        t2_npy = np.load(testA_npy_list[i].replace('/testA/', '/testB/'))
        
        axs[i, 0].imshow(t1_npy, cmap='gray')
        axs[i, 0].set_title("T1 MRI")
        axs[i, 0].axis("off")

        axs[i, 1].imshow(t2_npy, cmap='gray')
        axs[i, 1].set_title("T2 MRI")
        axs[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(npy_dir, 'samples.png'), dpi=100)
    return None


if __name__ == "__main__":
    # arguments for preprocessing files
    args = parser.parse_args()
    # create temporary directory
    temp_dir = tempfile.mkdtemp()
    print(f"Extracting zip file to temporary directory: {temp_dir}")
    # preprocess the data
    try:
        # extract the dataset files
        with zipfile.ZipFile(args.zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        # create npy data files directory
        npy_dir = os.path.join(args.dataset_root, args.dataset_name)
        print(f"Create npy data files directory: {npy_dir}")
        os.makedirs(npy_dir, exist_ok=True)
        print("Finish extracting data. Now process the NIFTI files to npy.")
        # preprocess data
        unc_nii_to_npy(temp_dir, npy_dir)
    finally:
        # clean up the temporary directory
        shutil.rmtree(temp_dir)
        print("Temporary directory removed.")

#!/bin/tcsh
#$ -o $HOME/prep_temp/dataset/logs/
#$ -e $HOME/prep_temp/dataset/logs/
#$ -q global
#$ -N dataset_job
#$ -l h_vmem=6G

# First unload any modules loaded by ~/.cshrc then load the defaults
module purge
module load nan/default
module load sge
# Load in script dependent modules here
module load ants/2.2.0

# set the working variables
# template and mask downloaded from http://nist.mni.mcgill.ca/?p=904 (ICBM 2009c Nonlinear Symmetric)
set ants_template = $HOME/prep_temp/dataset/mni_icbm152_t1_tal_nlin_sym_09c.nii
set ants_brain_mask = $HOME/prep_temp/dataset/mni_icbm152_t1_tal_nlin_sym_09c_mask.nii

setenv working_data $HOME/prep_temp/dataset
setenv sge_index ${working_data}/sge_index


# Search the file for the SGE_TASK_ID number as a line number
set file="`awk 'FNR==$SGE_TASK_ID' ${sge_index}`"

# Used the tcsh :t to find last part of path,
# then :r to remove .gz then :r to remove .nii
set file_name=${file:t:r:r}

# Based on https://github.com/ANTsX/ANTs/blob/master/Scripts/antsBrainExtraction.sh
# https://github.com/ntustison/BasicBrainMapping
# https://github.com/ntustison/antsBrainExtractionExample/blob/master/antsBrainExtractionCommand.sh
# https://github.com/ANTsX/ANTs/blob/master/Scripts/antsRegistrationSyN.sh
# https://sourceforge.net/p/advants/discussion/840261/thread/ca08a5aa74/?limit=25

bash antsBrainExtraction.sh \
  -d 3 \
  -a ${file} \
  -e ${ants_template} \
  -m ${ants_brain_mask} \
  -o ${working_data}/subjects_output/${file_name}_

bash antsRegistrationSyNQuick.sh \
#bash antsRegistrationSyN.sh \
  -d 3 \
  -f ${ants_template} \
  -m ${working_data}/subjects_output/${file_name}_BrainExtractionBrain.nii.gz \
  -x ${ants_brain_mask} \
  -t s \
  -o ${working_data}/subjects_output/${file_name}_


rm ${working_data}/subjects_output/${file_name}_InverseWarped.nii.gz
rm ${working_data}/subjects_output/${file_name}_BrainExtractionPrior0GenericAffine.mat
rm ${working_data}/subjects_output/${file_name}_BrainExtractionMask.nii.gz
rm ${working_data}/subjects_output/${file_name}_BrainExtractionBrain.nii.gz
rm ${working_data}/subjects_output/${file_name}_1Warp.nii.gz
rm ${working_data}/subjects_output/${file_name}_1InverseWarp.nii.gz
rm ${working_data}/subjects_output/${file_name}_0GenericAffine.mat

#!/bin/bash
#SBATCH --job-name=voidprofiles
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --time=00:02:00
#SBATCH --partition=debug
## An email is sent when the job is canceled or finished
## Change "root" for your e-mail address
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=fmcaporaso@mi.unc.edu.ar


. /etc/profile
##source $HOME/.bashrc

##cd /mnt/projects/lensing/HALO_SHAPE/MICEv1.0/codes_HS_micev1.0/

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_py39

##srun ./run_fit_zphot.sh
##srun ./run_fit_zphotw3.sh
srun python profiles/perfiles.py --NCORES 50 --NBINS 100 --Rv_min 6.0 --Rv_max 9.622 --z_min 0.2 --z_max 0.4 --filename 'a'
srun python profiles/perfiles.py --NCORES 50 --NBINS 100 --Rv_min 6.0 --Rv_max 9.622 --z_min 0.2 --z_max 0.4 --rho2_min 0.0 --rho2_max 100.0 --filename 'a'
srun python profiles/perfiles.py --NCORES 50 --NBINS 100 --Rv_min 6.0 --Rv_max 9.622 --z_min 0.2 --z_max 0.4 --rho2_min -1.0 --rho2_max 0.0 --filename 'a'
srun python profiles/perfiles.py --NCORES 50 --NBINS 100 --Rv_min 9.622 --Rv_max 50.0 --z_min 0.2 --z_max 0.4 --filename 'a'
srun python profiles/perfiles.py --NCORES 50 --NBINS 100 --Rv_min 9.622 --Rv_max 50.0 --z_min 0.2 --z_max 0.4 --rho2_min 0.0 --rho2_max 100.0 --filename 'a'
srun python profiles/perfiles.py --NCORES 50 --NBINS 100 --Rv_min 9.622 --Rv_max 50.0 --z_min 0.2 --z_max 0.4 --rho2_min -1.0 --rho2_max 0.0 --filename 'a'

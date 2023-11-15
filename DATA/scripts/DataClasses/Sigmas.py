"""
Class to add sigma_n density estimators to a table.

Arguments for initialization are:
    * name (string): Name of .csv file without extension. 
    * dir (string): Directory containing the files.
    * df_gal (pd.DataFrame or None): DataFrame with galaxies. If None, the table is read from the file. (default: None)

The main() method performs all the tasks. 
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from astropy.cosmology import FlatLambdaCDM
import math
import multiprocessing

vels = [1000, 3000, 5000, 10000]
neig = [3,5,7,10]

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

class Sigmas:

    def __init__(self, name: str, dir: str, df_gal: pd.DataFrame|None = None) -> None:
        self.name = name #name of files
        self.dir = dir #directory for reading and saving files
        self.df_gal = df_gal
        print(f'SIGMAS FOR {self.name}')

    def main(self, njobs = 1):
        self._read_files()
        self._sigmas_no_cuts()
        self._velocity_bins()
        self._sigmas_multiprocessing(njobs)
        self._save()
        return self.df_gal

    def _read_files(self):
        print('Reading files...')
        if self.df_gal is None: self.df_gal = pd.read_csv(f'{self.dir}clean_tables/{self.name}.csv')
        self.dat = pd.read_table(f'{self.dir}Wen+Han/{self.name}.dat', delim_whitespace=True, usecols=[0,1,16], names=['ra','dec','phot_z'])
        self.dat  = self.dat.drop_duplicates(subset = ['ra','dec'])
        self.mpc_deg = cosmo.kpc_proper_per_arcmin(self.df_gal['phot_z'].values).value / 1000 * 60  # Mpc/degree

    def _velocity_bins(self):
        # velocities
        c = 300_000 # km/s
        # z + 1 = sqrt((1 + v/c)/(1-v/c))
        vels_dat = np.array([(z**2 + z*2)/(z**2 + 2*z + 2) *c for z in self.dat['phot_z'].values])
        vels_df = np.array([(z**2 + z*2)/(z**2 + 2*z + 2) *c for z in self.df_gal['phot_z'].values] )
        
        for v in vels:
            for j in neig:
                self.df_gal = self.df_gal.assign(a = -99)
                self.df_gal = self.df_gal.rename({'a': f'sigma_{j}_{v}'}, axis = 1, errors = 'raise')

        # divide sample in velocity bins.
        # this will be the bins limits
        bins = np.linspace(min(vels_dat), stop= max(vels_dat), num = math.floor((max(vels_dat) - min(vels_dat))/10_000))
        # their labels
        self.labels = np.arange(0, len(bins)-1)
        # get an array with the bin of each galaxy
        bins_df = pd.cut(vels_df, bins=bins, labels= self.labels)
        bins_dat = pd.cut(vels_dat, bins=bins, labels= self.labels)
        # get a list of coords and vels and everything for each bin
        self.dfs, self.dats, self.vels_dats, self.vels_dfs, self.mpc_degs= [[] for i in range(5)]
        for i in self.labels:
            self.dfs.append(self.df_gal[bins_df == i])
            self.dats.append(self.dat[bins_dat == i])
            self.vels_dfs.append(vels_df[bins_df == i])
            self.vels_dats.append(vels_dat[bins_dat == i])
            self.mpc_degs.append(self.mpc_deg[bins_df == i])

    def _sigmas_multiprocessing(self, njobs = 1):
        if njobs == -1: njobs = None
        with multiprocessing.Pool(njobs) as p:
            dfs_sigma = p.map(self.sigmas_in_bin_velocity_cuts, self.labels)
        # without multiprocessing:
        # dfs_sigma = map(sigmaVels.bin_search, data.labels)

        self.df_gal = pd.concat(dfs_sigma).sort_values('ra')

    def sigmas_in_bin_velocity_cuts(self, l):
        # l is the bin label
        # get the correct df and dat bins...
        # search in this bin, and the two adjecent ones
        if l==0:
            coords_dat = pd.concat([self.dats[l], self.dats[l+1]])[['dec','ra']]
            vels_this_bin = np.concatenate([self.vels_dats[l], self.vels_dats[l+1]])
        elif l == self.labels[-1]:
            coords_dat = pd.concat([self.dats[l-1], self.dats[l]])[['dec','ra']]
            vels_this_bin = np.concatenate([self.vels_dats[l-1], self.vels_dats[l]])
        else:
            coords_dat = pd.concat([self.dats[l-1], self.dats[l], self.dats[l+1]])[['dec','ra']]
            vels_this_bin = np.concatenate([self.vels_dats[l-1], self.vels_dats[l], self.vels_dats[l+1]])
        
        df_ = self.dfs[l].copy()
        df_ = df_.reset_index(drop=True)
        coords_df = df_[['dec','ra']].values
        coords_dat = coords_dat.reset_index(drop = True)
        

        # start by looking for a small number of neighbors
        n_neighbors= 100
        # this should never exceed the sample size.
        n_neigbors_max = coords_dat.shape[0]

        success = [[] for i in vels] # save here the index of galaxies with successful computation, one list for each velocity
        while any(len(b) < coords_df.shape[0] for b in success):

            nbrs = NearestNeighbors(n_neighbors= n_neighbors, algorithm='ball_tree', n_jobs = 1, metric= 'haversine')
            nbrs.fit(np.deg2rad(coords_dat.values)) 

            for i,c in enumerate(coords_df): 
                
                if not all([i in b for b in success]): # if the galaxy has not succeded for all velocities
                    # find neighbors for galaxy
                    n_all = nbrs.kneighbors([np.deg2rad(c)], return_distance= True) 
                    
                    for vidx,v in enumerate(vels):
                        if not i in success[vidx]:
                            n_valid = []

                            # check if the first neighbor's velocity is addecuate, if not, pass to the next, until we have 11 neighbors.
                            for idx,dist in zip(n_all[1][0], n_all[0][0]):
                                if (abs(vels_this_bin[idx] - self.vels_dfs[l][i]) < v ):
                                    n_valid.append(dist)

                                    if len(n_valid) >= 11:
                                        break
                            
                            # if we do get 11 neighbors, then this is a success and we save the sigmas
                            if len(n_valid) >= 11:
                                success[vidx].append(i)
                                for j in neig:
                                    ang_dist = np.rad2deg(n_valid[j])
                                    dist = ang_dist * self.mpc_degs[l][i]
                                    df_.at[i, f'sigma_{j}_{v}'] = float(j)/ np.pi /dist**2
                            # if we do not have all the neighbors, but we have already searched the whole sample, save what we can
                            elif n_neighbors == n_neigbors_max:
                                for j in neig:
                                    try:
                                        ang_dist = n_valid[j] *180 / np.pi
                                        dist = ang_dist * self.mpc_deg[i]
                                        df_.at[i, f'sigma_{j}_{v}'] = float(j)/ np.pi /dist**2
                                    except:
                                        pass

            # print some info:
            print(f'bin {l}, n_neighbors: {n_neighbors}, successful: {[len(b) for b in success]}')
                    
            # do everything again, with a larger n_neighbors.
            if n_neighbors < n_neigbors_max:
                n_neighbors *= 5
                if n_neighbors > n_neigbors_max:
                    n_neighbors = n_neigbors_max
            else:
                break
        print(f'Finish bin {l}')
        return df_
    
    def _sigmas_no_cuts(self):
        nbrs = NearestNeighbors(n_neighbors= 11, algorithm='ball_tree', n_jobs = -1, metric= 'haversine') # the first neighbor is the point itself
        nbrs.fit(np.deg2rad(self.dat[['dec', 'ra']].values)) # fit it to the whole sample
        n_all = nbrs.kneighbors(np.deg2rad(self.df_gal[['dec', 'ra']].values), return_distance= True)  # find neighbors only for my sample
        for i in neig:
            ang_dist = np.rad2deg(n_all[0][:,i])  
            dist = ang_dist * self.mpc_deg
            self.df_gal[f'sigma_{i}'] = i/ np.pi /dist**2

    def _save(self):
        print('Saving...')
        self.df_gal.drop_duplicates(subset = ['ra','dec'])
        self.df_gal.to_csv(f'{self.dir}clean_tables/{self.name}.csv', index= False)

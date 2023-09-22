"""
Add the following features to a table:  
    * 2d sigma_3,5,7,10
    * 2d sigma_3,5,7,10, with velocity cuts

From command line: python add_features.py  HSC-unWISE-W01.csv HSC-unWISE-W01.dat
Arguments: 
    csv table with galaxies.
    dat table with all objects (published by Wen & Han)
"""

import pandas as pd
import numpy as np
import sys
from sklearn.neighbors import NearestNeighbors
from astropy.cosmology import FlatLambdaCDM
import math
import time
import multiprocessing
# import os

vels = [1000, 3000, 5000, 10000]
neig = [3,5,7,10]

# for debugging....
# path = os.path.dirname(__file__) +'/'  #of this file
# df_file = path + 'clean-HSC-unWISE-W01.csv'
# dat_file = path + 'HSC-unWISE-W01.dat'

df_file = sys.argv[1] 
dat_file = sys.argv[2] 

class Data:
    def __init__(self) -> None:
        self.read()
        self.cosmo()
        
    def read(self):
        self.df = pd.read_csv(df_file)
        self.dat = pd.read_table(dat_file, delim_whitespace=True, usecols=[0,1,16], names=['ra','dec','phot_z'])
        self.dat  = self.dat.drop_duplicates(subset = ['ra','dec'])
    
    def cosmo(self):
        lcdm = FlatLambdaCDM(H0=70, Om0=0.3)
        self.mpc_deg = lcdm.kpc_proper_per_arcmin(self.df['phot_z'].values).value / 1000 * 60  # Mpc/degree

    def velocities(self):
        # velocities
        c = 300_000 # km/s
        # z + 1 = sqrt((1 + v/c)/(1-v/c))
        vels_dat = np.array([(z**2 + z*2)/(z**2 + 2*z + 2) *c for z in self.dat['phot_z'].values])
        vels_df = np.array([(z**2 + z*2)/(z**2 + 2*z + 2) *c for z in self.df['phot_z'].values] )
        
        for v in vels:
            for j in neig:
                self.df = self.df.assign(a = -99)
                self.df = self.df.rename({'a': f'sigma_{j}_{v}'}, axis = 1, errors = 'raise')

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
            self.dfs.append(self.df[bins_df == i])
            self.dats.append(self.dat[bins_dat == i])
            self.vels_dfs.append(vels_df[bins_df == i])
            self.vels_dats.append(vels_dat[bins_dat == i])
            self.mpc_degs.append(self.mpc_deg[bins_df == i])
        
class SigmaWithVelocitieCut:
    
    def __init__(self, data: Data) -> None:
        self.data = data
        self.data.velocities()

    def bin_search(self, l):
        # l is the bin label
        # get the correct df and dat bins....
        # search in this bin, and the two adjecent ones
        if l==0:
            coords_dat = pd.concat([self.data.dats[l], self.data.dats[l+1]])[['dec','ra']]
            vels_this_bin = np.concatenate([self.data.vels_dats[l], self.data.vels_dats[l+1]])
        elif l == self.data.labels[-1]:
            coords_dat = pd.concat([self.data.dats[l-1], self.data.dats[l]])[['dec','ra']]
            vels_this_bin = np.concatenate([self.data.vels_dats[l-1], self.data.vels_dats[l]])
        else:
            coords_dat = pd.concat([self.data.dats[l-1], self.data.dats[l], self.data.dats[l+1]])[['dec','ra']]
            vels_this_bin = np.concatenate([self.data.vels_dats[l-1], self.data.vels_dats[l], self.data.vels_dats[l+1]])
        
        df_ = self.data.dfs[l].copy()
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
                                if (abs(vels_this_bin[idx] - self.data.vels_dfs[l][i]) < v ):
                                    n_valid.append(dist)

                                    if len(n_valid) >= 11:
                                        break
                            
                            # if we do get 11 neighbors, then this is a success and we save the sigmas
                            if len(n_valid) >= 11:
                                success[vidx].append(i)
                                for j in neig:
                                    ang_dist = np.rad2deg(n_valid[j])
                                    dist = ang_dist * self.data.mpc_degs[l][i]
                                    df_.at[i, f'sigma_{j}_{v}'] = float(j)/ np.pi /dist**2
                            # if we do not have all the neighbors, but we have already searched the whole sample, save what we can
                            elif n_neighbors == n_neigbors_max:
                                for j in neig:
                                    try:
                                        ang_dist = n_valid[j] *180 / np.pi
                                        dist = ang_dist * self.data.mpc_deg[i]
                                        df_.at[i, f'sigma_{j}_{v}'] = float(j)/ np.pi /dist**2
                                    except:
                                        pass

            # print some info:
            print(f'bin {l}, n_neighbors: {n_neighbors}, successful: {[len(b) for b in success]}')
                    
            # do everything again, with a larger n_neighbors.
            if n_neighbors < n_neigbors_max:
                n_neighbors *= 10
                if n_neighbors > n_neigbors_max:
                    n_neighbors = n_neigbors_max
            else:
                break
        print(f'Finish bin {l}')
        return df_

class Sigma2D:

    def __init__(self, data:Data) -> None:
        self.data = data

    def search(self):
        nbrs = NearestNeighbors(n_neighbors= 11, algorithm='ball_tree', n_jobs = -1, metric= 'haversine') # the first neighbor is the point itself
        nbrs.fit(np.deg2rad(self.data.dat[['dec', 'ra']].values)) # fit it to the whole sample
        n_all = nbrs.kneighbors(np.deg2rad(self.data.df[['dec', 'ra']].values), return_distance= True)  # find neighbors only for my sample
        for i in neig:
            ang_dist = np.rad2deg(n_all[0][:,i])  
            dist = ang_dist * self.data.mpc_deg
            self.data.df[f'sigma_{i}'] = i/ np.pi /dist**2


if __name__ == '__main__':
    start = time.time()
    
    data = Data()

    print('2D sigmas...')
    sigma2d = Sigma2D(data = data)
    sigma2d.search()


    print('2D sigmas with velocity cuts')
    sigmaVels = SigmaWithVelocitieCut(data= data)
    with multiprocessing.Pool() as p:
        dfs_sigma = p.map(sigmaVels.bin_search, data.labels)

    # without multiprocessing:
    # dfs_sigma = map(sigmaVels.bin_search, data.labels)

    df_final = pd.concat(dfs_sigma).sort_values('ra')
    end = time.time()
    print(f'Time taken: {end - start} s')

    print('Saving...')
    df_final.drop_duplicates()
    df_final.to_csv('sigma_' + df_file, index= False)
import os
from DataClasses.FeatureRetriever import FeatureRetriever
from DataClasses.IdRetriever import IdRetriever
from DataClasses.DataCleaner import  DataCleaner
from DataClasses.NewFeatures import NewFeatures
# from DataClasses.Kcorrections import Kcorrections
from DataClasses.Sigmas import Sigmas
from DataClasses.RedshiftLimits import RedshiftLimits
from DataClasses.SampleSplit import SampleSplit
import time

if __name__ == '__main__':
    names = ['HSC-unWISE-W01', 'HSC-unWISE-W02', 'HSC-unWISE-W03', 'HSC-unWISE-W04', 'HSC-unWISE-W05', 'HSC-unWISE-W06']
    dir = os.path.dirname(__file__) +'/'  #of this file

    for name in names:
        df_gal = None
        # idRet = IdRetriever(name, dir, 'gimenam144')
        # df_gal = idRet.main(write_query= True, run_query= True)

        # featRet = FeatureRetriever(name, dir, 'gimenam144', df_gal= df_gal)
        # df_gal = featRet.main(write_query= True, run_query = True)

        # datClean = DataCleaner(name, dir, df_gal= df_gal)
        # df_gal = datClean.main()

        # newFeat = NewFeatures(name, dir, df_gal= df_gal)
        # df_gal = newFeat.main(njobs= 6)

        # # kCorr = Kcorrections(name, dir, df_gal= df_gal)
        # # df_gal = kCorr.main()

        # sigma = Sigmas(name, dir, df_gal= df_gal)
        # df_gal = sigma.main(njobs = 6)

        # zLims = RedshiftLimits(name, dir, df_gal= df_gal)
        # df_gal = zLims.main(njobs = 6)

        split = SampleSplit(name, dir, df_gal= df_gal)
        df_gal = split.main()


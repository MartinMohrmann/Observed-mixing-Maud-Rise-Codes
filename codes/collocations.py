# Fully working version to create collocation datasets for era5/floats
import utils
import numpy
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
import cmocean
import pandas
import datetime
import netCDF4
from matplotlib import ticker
import gsw
import xarray
import importlib
import glob
import os # to be able to reload user modules with importlib.reload(utils)


maxdate = datetime.datetime(2021,5,11)#datetime.datetime(2020,12,31)
mindate = datetime.datetime(1970,1,1)

def collocate(lat, lon, datetimes):
    oldmonth, oldyear = 0,0
    rows = []
    counter = 0

    # --- Topographic collocation
    lat_grid_sithick = xarray.open_dataset('../data/ice_thickness/south_lat_12km.nc')['Latitude [degrees]']
    lon_grid_sithick = xarray.open_dataset('../data/ice_thickness/south_lon_12km.nc')['Longitude [degrees]']
    etopofile = netCDF4.Dataset('../data/etopo1.nc')
    elevation = etopofile.variables['Band1'][:]
    elevation_gradx = numpy.gradient(etopofile.variables['Band1'][:], axis=0)
    elevation_grady = numpy.gradient(etopofile.variables['Band1'][:], axis=1)
    dftopo = xarray.DataArray(data=elevation, dims=["lat", "lon"], 
                coords=[etopofile.variables['lat'][:],etopofile.variables['lon'][:]])
    dftopogradx = xarray.DataArray(data=elevation_gradx, dims=["lat", "lon"], 
                coords=[etopofile.variables['lat'][:],etopofile.variables['lon'][:]])
    dftopogrady = xarray.DataArray(data=elevation_grady, dims=["lat", "lon"], 
                coords=[etopofile.variables['lat'][:],etopofile.variables['lon'][:]])
    # --- end topographic collocation

    for index, dtime in enumerate(datetimes):
        month = str(dtime.month).zfill(2)
        year = dtime.year
        day = str(dtime.day).zfill(2)
        # print(year, month, day)
        if (mindate >= datetime.datetime(dtime.year, dtime.month, dtime.day)):
            # This exists mainly for testing purpose
            continue
        if (maxdate <= datetime.datetime(dtime.year, dtime.month, dtime.day)):
            print('collocation reached specified end date.')
            break
        if (oldmonth != month or oldyear != year):
            print('reloading... %s-%s'%(month, year))
            if os.path.isfile('../data/era5/download_%s_%s.nc'%(year,month)):
                df = xarray.open_dataset('../data/era5/download_%s_%s.nc'%(year,month))
            else:
                print('No Era5 data downloaded for %s_%s.nc'%(year,month))
                continue
        df['rotw10'] = df['v10'].differentiate('longitude') - df['u10'].differentiate('latitude')

        # SEA ICE THICKNESS CORRELATION START
        filelist = glob.glob('../data/ice_thickness/%s%s%s_hvsouth_rfi_l1c.nc'%(year, month, day))
        if filelist:
            dataset = xarray.open_mfdataset(filelist, concat_dim='time')
            dataset['lat'] = lat_grid_sithick#xarray.open_dataset('../data/ice_thickness/south_lat_12km.nc')['Latitude [degrees]']
            dataset['lon'] = lon_grid_sithick#xarray.open_dataset('../data/ice_thickness/south_lon_12km.nc')['Longitude [degrees]']
            dataset = dataset.set_coords(['lat','lon'])

            # First, find the index of the grid point nearest a specific lat/lon.   
            abslat = numpy.abs(dataset.lat-lat[index])
            abslon = numpy.abs(dataset.lon-lon[index])
            c = numpy.maximum(abslon, abslat)
            A = numpy.where(c == numpy.min(c))

            if numpy.shape(A)[1] == 1:
                ([xloc], [yloc]) = A
            else:
                # very unusual/unlikely case that float coordinate lies EXACTLY on grid cell border
                # In that case, nearest grid point algorithm returns two values, of which we take the first one
                xloc, yloc = (A[0][0], A[1][0])
                print('gridbordercase')

            # Now I can use that index location to get the values at the x/y diminsion
            point_dataset = dataset['thickness'].isel(X=xloc, Y=yloc)
            df['sithick'] = point_dataset.values
        else:
            df['sithick'] = None
        # SEA ICE THICKNESS CORRELATION END

        # ETOPO CORRELATION START
        df['depth_topo'] = dftopo.interp(lon=lon[index], lat=lat[index])
        df['depth_topo_gradx'] = dftopogradx.interp(lon=lon[index], lat=lat[index])
        df['depth_topo_grady'] = dftopogrady.interp(lon=lon[index], lat=lat[index])

        # Actually, all the ERA5 collocation happens here
        row = df.interp(longitude=lon[index], latitude=lat[index], time=dtime)
        rows.append(row)
        oldmonth, oldyear = month, year
        counter += 1
    return rows

def collocate_my_floats():
    for floatid in [
        '300234067208900', 
        '300234068638900']:
        coord = pandas.read_pickle('../data/coordinates_%s.pkl'%floatid)
        coord_filtered = coord[coord.CEP<=20]
        coord_filtered = coord_filtered.set_index('date').resample('1h').mean()
        coord_filtered = coord_filtered.interpolate(method='linear')
        rows = collocate(coord_filtered['latitude'], coord_filtered['longitude'], coord_filtered.index)

        df_old_era4 = xarray.concat(rows, dim='time')
        try:
            df_old_era4.to_netcdf('../data/collocation_%s.nc'%floatid, mode='w')
        except:
            print('file writing exception, I forgot to delete previous file')
            breakpoint()

def collocate_ARGO_floats():
    names = [
        '5904468_Mprof.nc', 
        #'5905382_Mprof.nc',  same as 'GL_PR_PF_5905382.nc'
        'GL_PR_PF_5903616.nc', 
        #'GL_PR_PF_7900123.nc', missing winter
        '5904471_Mprof.nc', 
        #'GL_PR_PF_1901903.nc', missing winter
        'GL_PR_PF_5905382.nc', 
        #'GL_PR_PF_7900640.nc' not so relevant
        ]
    for name in names:
        df = xarray.open_dataset('../data/SOCCOM/%s'%name)
        if 'JULD' in df.keys():
            key = 'JULD'
        else:
            key = 'TIME'

        times = df[key]
        latitudes = df['LATITUDE']
        longitudes = df['LONGITUDE']
        df = pandas.DataFrame.from_dict(dict(date=times, lon=longitudes, lat=latitudes))
        df = df.set_index('date').resample('1h').mean()
        df = df.interpolate(method='linear')
        rows = collocate(df['lat'], df['lon'], df.index)
        if not rows:
            print('No matching collocation for %s'%name)
            continue
        df_col = xarray.concat(rows, dim='time')
        try:
            df_col.to_netcdf('../data/collocation_%s.nc'%name.split('.')[0], mode='w')
        except:
            print('file writing exception, I forgot to delete previous file')
            breakpoint()

if __name__ == "__main__":
    # The to_netcdf-function does not seem to overwrite old files automatically 
    # In case of error while writing, delete outdated '../data/collocation_%s.nc' outputs first
    collocate_my_floats() # This would be our Maud Rise Float and Reference Float
    # collocate_ARGO_floats() # Soccom and Argo Floats
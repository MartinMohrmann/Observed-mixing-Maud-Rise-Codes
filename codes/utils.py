import imaplib
import base64
import os
import email
import datetime
import numpy
import gsw
import pandas
import cmocean
from scipy.interpolate import griddata
import re
import math
import netCDF4
import matplotlib
import matplotlib.dates as mdates
import xarray
import warnings
import locale
import scipy
import copy

mld_density_diff=0.02
monthDict={1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}


# The float data files as processed with the NKE processing tools
# These are the columns 
# 0 always empty
# 1 hours
# 2 minutes
# 3 seconds
# 4 pressure (db)
# 5 temperature (°C)
# 6 salinity (PSU)
NKE_MaudRise = numpy.genfromtxt("../data/converted/old/Ascent profile CTD Message.csv", encoding="cp1252", delimiter=";", usecols=[1,2,5,6,7],skip_header=1)
NKE_Compare  = numpy.genfromtxt("../data/converted/new/Ascent profile CTD Message.csv", encoding="cp1252", delimiter=";", usecols=[1,2,5,6,7],skip_header=1)


def download_floatdata(floatid):
    """This function downloads the SBD files directly from the GU mailserver

    Parameters: 
        floatid (str): either 300234068638900 or 300234067208900
    
    Returns:
        None (but saves the files)"""

    email_user = secrets.user
    email_pass = secrets.pass

    mail = imaplib.IMAP4_SSL("outlook.office365.com",993)
    mail.login(email_user, email_pass)

    index = 0

    for searchbox in [floatid]:#'Inbox', floatid]: # This is the email folders where the data arrives
        print('searching in %s only'%searchbox)
        mail.select(floatid) # change 'Inbox' to floatid to search explicitly in my postboxes, but only Feb 2020 onwards :(

        #type, data = mail.search(None, '(HEADER Subject "SBD Msg From Unit: %s")'%floatid)      #FROM "tech163@fusionswift.com"  '(HEADER Subject "Subject Here")'
        type, data = mail.search(None, 'ALL')
        mail_ids = data[0]
        id_list = mail_ids.split()
        print('found %s mails in %s'%(len(id_list), searchbox))

        for num in data[0].split():
            index = index + 1
            typ, data = mail.fetch(num, '(RFC822)' )
            raw_email = data[0][1]  # converts byte literal to string removing b''
            raw_email_string = raw_email.decode('utf-8')
            email_message = email.message_from_string(raw_email_string)  # downloading attachments
            for part in email_message.walk():
                # this part comes from the snipped I don't understand yet...
                if part.get_content_maintype() == 'multipart':
                    continue
                if part.get('Content-Disposition') is None:
                    continue
                fileName = part.get_filename()
                if bool(fileName):
                    if floatid == '300234068638900':
                        filePath = os.path.join('../data/sbd/new/', fileName)
                    else:
                        filePath = os.path.join('../data/sbd/old/', fileName)
                    if not os.path.isfile(filePath) :
                        fp = open(filePath, 'wb')
                        fp.write(part.get_payload(decode=True))
                        fp.close()
                    subject = str(email_message).split("Subject: ", 1)[1].split("\nTo:", 1)[0]
    print("processed %s mails, existing files were not renewed"%index)


def create_coordinates_dates_list(floatid):
    """ Parses all the mails on the GU server for the float coordinates, dates and CEP values
    
    Parameters: 
        floatid (str): either 300234068638900 or 300234067208900
    
    Returns:
        None (but saves a pandas dataframe)"""
    email_user = secrets.user
    email_pass = secrets.pass

    mail = imaplib.IMAP4_SSL("outlook.office365.com",993)
    mail.login(email_user, email_pass)

    floatid = floatid # 300234067208900 # 300234068638900

    lats = []
    lons = []
    CEP  = []
    mdat = []

    for folder in ['Inbox', floatid]:
        mail.select(300234067208900) # 300234067208900 # 300234068638900

        type, data = mail.search(None, '(HEADER Subject "SBD Msg From Unit: %s")'%floatid)
        mail_ids = data[0]
        id_list = mail_ids.split()
        index = 0

        for num in data[0].split():
            latlon = None # making sure now data from previous iteration is reused
            mCEP = None
            mydate = None
            index = index + 1
            typ, data = mail.fetch(num, '(RFC822)' )
            raw_email = data[0][1]# converts byte literal to string removing b''
            raw_email_string = raw_email.decode('utf-8')
            #email_message = email.message_from_string(raw_email_string)# downloading attachments
            linelist = raw_email_string.split('\n')
            for line in linelist:
                if 'Unit Location' in line:
                    latlon = re.findall('[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?', line)
                if 'CEPradius' in line:
                    mCEP = int(re.findall('[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?', line)[0])
                if 'Time of Session' in line:
                    # breakpoint()
                    locale.setlocale(locale.LC_TIME,'en_GB.utf8') # This is important to read in the date
                    # in the right format with computers running on non-english locale
                    mydate = datetime.datetime.strptime(line[-25:-1], "%a %b %d %H:%M:%S %Y")
                    #print('worked!', line[-21:-1])
                    #except:
                    #    print('failed!', line[-21:-1])
                    #    import pdb; pdb.set_trace()
            if (latlon and mydate and mCEP):
                lons.append(float(latlon[1]))
                lats.append(float(latlon[0]))
                mdat.append(mydate)
                CEP.append(mCEP)
            else:
                print('Hoppla, für diese mail fehlte eine der variablen...')

    df = pandas.DataFrame()
    df['date'] = mdat
    df['longitude'] = lons
    df['latitude'] = lats
    df['CEP'] = CEP
    if floatid == '300234068638900':
        # filtering out on-deck configuration data
        NKE_float = filter_parking(NKE_Compare)
        startdate = datetime.datetime(2020,2,17,12,0,0)
    elif floatid == '300234067208900':
        NKE_float = filter_parking(NKE_MaudRise)
        startdate = datetime.datetime(2018,12,17,12,0,0)
    df = df[df['date']>startdate]
    df.to_pickle('../data/coordinates_%s.pkl'%floatid)


def filter_parking(NKE_float):
    """ Filter the values of the float in parking position, 
    since they are not processed with the profiles information

    Parameters: 
        NKE_float file as directly imported from the NKE-tool output

    Returns:
        NKE_float arrays without the parking lines"""
    filter = []
    # filter out the parking readings
    for i,line in enumerate(NKE_float):
        if line[2] == 1000:
            filter.append(i)
        else:
            continue
    # breakpoint()
    NKE_float = numpy.delete(NKE_float, filter, 0)
    print('... done filtering out parking readings, length now: %s'%(len(NKE_float)))
    return NKE_float


def repair_hours(NKE_float):
    """ Add the time column 'hour' to every line of the dataset, 
    which is only provided every n steps in the original dataset 

    Parameters: 
        NKE_float (numpy.array) after filtering out the parking values

    Returns:
        NKE_float array with consistent time column """
    for i in range(0,len(NKE_float[:,1])):
        if numpy.isnan(NKE_float[i,1]):
            NKE_float[i,1] = hour
            NKE_float[i,0] = cycle
        else:
            hour = NKE_float[i,1]
            cycle = NKE_float[i,0]
    return NKE_float


def add_timeaxis(NKE_float, pfloat):
    """ Convert the time from the format 'hours since deployment into' a python datetime object

    Parameters:
        NKE_float (numpy.array): Float data after filtering, and adding the hours
        pfloat (str): Either 'new' or 'old'

    Returns:
        dates (list of datetimes): dates for each float measurement
        timestamps (list of UNIX milliseconds): timestamps for easier linear plotting and interpolation,
         because e.g. griddata routines don't accept dates as interpolation axis.
    """ 
    dates = []
    timestamps = []
    print(len(NKE_float[:]))
    for i in range(0,len(NKE_float[:])):
        if pfloat in ['old', '300234067208900.nc']:
            date = datetime.datetime(year=2018, month=12, day=16) + datetime.timedelta(hours=NKE_float[i])
        else:
            date = datetime.datetime(year=2020, month=2, day=16) + datetime.timedelta(hours=NKE_float[i])
        dates.append(date)
        timestamps.append(date.timestamp())
    return dates, timestamps


def sort_missing(NKE_float):
    import gamman as gn

    """filtering out data rows where one or more sensors did not return a values
    Note: it is a bit sad to filter out this additional readings, but with incomplete
    data sets computation of e.g. potential temperature is impossible, as well as most derived values"""
    # curently this is not filtering out anything, because parking readings are filtered out elsewhere
    # Gradient values like N² should not be computed here, because (or if) the order of values is still chaotic, 
    # the sorting routine comes later
    counter = 0
    data = dict()#sal=[], tem=[], den=[], pre=[], tim=[], cyk=[], nsq=[], bvf=[])
    #for i in range (0,len(NKE_float[:,1])-1):
        # if numpy.isnan([NKE_float[i,1], NKE_float[i,2], NKE_float[i,3], NKE_float[i,4]]).any(): # alternative
        #if any([elem=='' for elem in [NKE_float[i,1], NKE_float[i,2], NKE_float[i,3], NKE_float[i,4]]]):
            # This would indicate some data is missing for the reading
            #print('data missing')
            #counter += 1
            #continue
            # I can consider checking if my pressure has any offset, e.g. surface pressure yes no?
    data['sal'] = gsw.SA_from_SP(NKE_float[:,4], NKE_float[:,2], 3, -65)
    # data['sal'] = NKE_float[:,4] #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!should be changed to asolute salinity, also for SOCCOM floats
    tem = gsw.CT_from_t(data['sal'], NKE_float[:,3], NKE_float[:,2])
    data['tem'] = tem #gsw.CT_from_t(data['sal'], NKE_float[:,3], NKE_float[:,2])
    data['den'] = gsw.density.sigma0(data['sal'], tem)
    data['pre'] = -NKE_float[:,2]
    data['tim'] = NKE_float[:,1]
    data['cyk'] = NKE_float[:,0].astype(int)
    data['rho'] = gsw.density.rho(data['sal'], data['tem'], NKE_float[:,2])# in situ density...
    #data['gamman'], gmin, gmax = gamma_n, dgl, dgh = gn.gamma_n(
    #    NKE_float[:,4], 
    #    NKE_float[:,3], 
    #    NKE_float[:,2],
    #    len(NKE_float[:,4]),3,-64)
    # breakpoint()
    #data['nsq'] = numpy.gradient(data['den'])/numpy.gradient(data['pre'])
    #data['bvf'] = (gsw.density.rho(NKE_float[i+1,4], NKE_float[i+1,3], NKE_float[i+1,2])-
    #               gsw.density.rho(NKE_float[i,4], NKE_float[i,3], NKE_float[i,2]))/
    #              (NKE_float[i+1,2]-NKE_float[i,2])

    print('... done sorting out incomplete entries, filtered out %s from %s entries...'%(counter, len(NKE_float[:,3])))
    return data


def create_dataframe(pfloat, force_reconstruction, customstartdate=None, customenddate=None):
    """ This is basically just a shortcut for using all of the above functions
    in a usefull order

    Parameters:
        pfloat (str): Either 'new' or 'old'

    Returns:
        df (pandas.dataframe): Useful dataframe of one float, with dates"""
    profiles = []
    filepath = '../data/CTDdata_%s.pkl'%pfloat
    if os.path.isfile(filepath) and not force_reconstruction:
        mtime = os.stat(filepath).st_mtime
        #if datetime.date.fromtimestamp(mtime) == datetime.date.today():
        df = pandas.read_pickle(filepath)
        #df = df.drop_duplicates(['pre', 'sal', 'tem']) 
        print('Returning previously computed DataFrame...')

        for i in range(0,max(df['cyk'])+1):
            df2 = df[df['cyk']==i]
            df2 = df2.drop_duplicates(subset = ["sal","tem","pre"])
            if len(df2)<=1:
                #print('filtered out empty profile i=%s'%i)
                pass
            else:
                nsquared, pmid = gsw.Nsquared(df2['sal'], df2['tem'], -df2['pre'])
                nsquared = numpy.append(nsquared, nsquared[-1])
                with pandas.option_context('mode.chained_assignment', None):
                    df2['nsq'] = nsquared
                #df2['sal'] = gsw.SA_from_SP(df2['sal'].values,df2['tem'].values, 3,-65)
                profiles.append(df2)

        return df, profiles
        #else:
        #    print('intermediate file outdated, reconstructing...')
        
    # order of functions is essential, don't reorder!
    if pfloat in ['new', '300234068638900.nc']:
        NKE_float = filter_parking(NKE_Compare)
        startdate = datetime.datetime(2020,2,17,12,0,0)
    elif pfloat in ['old', '300234067208900.nc']:
        NKE_float = filter_parking(NKE_MaudRise)
        startdate = datetime.datetime(2018,12,17,12,0,0)
    #elif pfloat == 'soc':
    else:
        df = create_soccom_dataframe(pfloat)
        #df.to_pickle(filepath)
        #return df

    if pfloat in ['new', '300234068638900.nc', 'old', '300234067208900.nc']:
        NKE_float = repair_hours(NKE_float)
        data = sort_missing(NKE_float)
        #data['nsq'] = numpy.append(data['nsq'], data['nsq'][-1]) # too early, values not sorted yet!
        df = pandas.DataFrame.from_dict(data)
        dates, timestamps = add_timeaxis(data['tim'], pfloat)
        df['dates'] = dates
        df['timestamps'] = timestamps
        # regular measurements for the new float start in the evening of the 17th Feb 2020/17th Dec 2018,
        # values recorded earlier are test-measurements.
        df = df[df['dates']>startdate]

        if customstartdate:
            # customstartdate=datetime.datetime(2020,4,1)
            df = df[df['dates']>customstartdate]
        if customenddate:
            # customenddate=datetime.datetime(2021,4,1)
            df = df[df['dates']<customenddate]

        df = df.sort_values(by=['cyk'])
        df = df.sort_values(by='pre', ascending=False)
        df = df.reset_index()

    #nsquared, pmid = gsw.Nsquared(df['sal'], df['tem'], df['pre'])
    #nsquared = numpy.append(nsquared, nsquared[-1])
    #df['nsq'] = nsquared
    #df['nsq'] = numpy.gradient(df['den'])/numpy.gradient(df['pre'])
    #df = df.drop_duplicates(['pre', 'sal', 'tem'])
    df.to_pickle(filepath)
    for i in range(0,max(df['cyk'])+1):
        df2 = df[df['cyk']==i]
        df2 = df2.drop_duplicates(subset = ["sal","tem","pre"])
        # df2 = df2.reset_index()
        # breakpoint()
        if len(df2)<=1:
            pass
            #print('filtered out empty profile i=%s'%i)
        else:
            # This solution is very ugly, since I am effectively shifting the pressure grid by one grid point
            # I should rather have the BVF in an extra array and treat it on pmid all the time!!!
            # print(len(df2))
            nsquared, pmid = gsw.Nsquared(df2['sal'], df2['tem'], -df2['pre'])
            # turner, gsw.stability.Turner_Rsubrho(SA, CT, p, axis=0)
            nsquared = numpy.append(nsquared, nsquared[-1])
            df2['nsq'] = nsquared
            #data['nsq'].append(nsquared[j])
            profiles.append(df2)

    return df, profiles


def create_soccom_dataframe(filename): # take WMOid as input instead
    """ This function takes the list of profiles as provided in the .nc files and reforms 
    it to a more 1d stream like format """

    # The following section is an alternative simpler implmentation with the
    # help of argopy, however I find it to not work as quick and precise as my own stuff
    """
    WMOid = 5904471
    import argopy
    from argopy import DataFetcher as ArgoDataFetcher
    ds = ArgoDataFetcher().float(WMOid).to_xarray()
    ds.argo.teos10(['SA', 'CT', 'SIG0', 'N2', 'PV', 'PTEMP'])
    if 'JULD' in ds.variables.keys():
        key = 'JULD'
    else:
        key = 'TIME'
    juldates = ds.variables[key][:]
    ts = (juldates - numpy.datetime64('1970-01-01T00:00:00Z')) / numpy.timedelta64(1, 's')
    ds['timestamps'] = ts
    print(ts)

    ds['pre'] = -ds['PRES']
    ds['sal'] = ds['PSAL']
    ds['tem'] = ds['TEMP']
    ds['nsq'] = ds['N2']
    ds['den'] = ds['SIG0']
    return ds
    """

    file = netCDF4.Dataset('../data/SOCCOM/%s'%filename)
    #len(file.variables['JULD_LOCATION'][:])
    # print('SOCCOM dataset end specified earlier manually %s'%end)
    normdates = []
    # Time key has different names for different floats and downloads
    if 'JULD' in file.variables.keys():
        key = 'JULD'
    else:
        key = 'TIME'
    juldates = file.variables[key][:]
    end = len(file.variables[key][:]) #5
    for mydate in juldates:
        # DATES (CAN BE) ALTERED TO OVERLAY WITH MY FLOAT DATA; ORIGINAL IS 1950, spatial overlay would be 1954
        normdates.append(datetime.datetime(1950, 1, 1) + datetime.timedelta(int(mydate) - 1))
    labels = [normdate.date() for normdate in normdates]
    data = dict(sal=[], tem=[], pre=[], tim=[], timestamps=[], den=[], lat=[], lon=[], nsq=[], cyk=[], dates=[], oxy=[]) 
    # Processing single points for later griddata
    for i in range (0,end):
    	# end must be the number of profiles here
        godkeys = [b'A', b'B']
        # for some floats, PROFILE_QCs do not exist :(
        if 'PROFILE_TEMP_QC' in file.variables.keys():
            if not (numpy.isin(file['PROFILE_TEMP_QC'][i], [b'A', b'B']) & 
                    numpy.isin(file['PROFILE_PSAL_QC'][i], [b'A', b'B']) & 
                    numpy.isin(file['PROFILE_PRES_QC'][i], [b'A', b'B'])):
                # print('bad quality flag for profile found, continue with next profile...')
                continue

        # df['PRES_ADJUSTED'][i] = numpy.nan = df['PRES_ADJUSTED_QC'][i], df['TEMP_ADJUSTED_QC'][i], df['PSAL_ADJUSTED_QC'][i]
        # df['DOXY_ADJUSTED_QC'][i]
        # abssal = file.variables['PSAL_ADJUSTED'][i]#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!should be changed to asolute salinity, also for SOCCOM floats
        abssal = gsw.SA_from_SP(file.variables['PSAL_ADJUSTED'][i], file.variables['PRES_ADJUSTED'][i], 3, -65)
        tcons = gsw.CT_from_t(abssal, file.variables['TEMP_ADJUSTED'][i], file.variables['PRES_ADJUSTED'][i])
        sigma0 = gsw.density.sigma0(abssal, tcons)
        nsquared, pmid = gsw.Nsquared(abssal, tcons, file.variables['PRES_ADJUSTED'][i])
        # This solution is very ugly, since I am effectively shifting the pressure grid by one grid point
        # I should rather have the BVF in an extra array and treat it on pmid all the time!!!
        nsquared = numpy.append(nsquared, nsquared[-1])

        for j in range (0, len(file.variables['PRES_ADJUSTED'][i])):
        	# j must be the running depth iteger here. 
            if numpy.ma.is_masked(file.variables['PRES_ADJUSTED'][i][j]):
                continue
            if numpy.ma.is_masked(file.variables['TEMP_ADJUSTED'][i][j]):
                continue
            if numpy.ma.is_masked(file.variables['PSAL_ADJUSTED'][i][j]):
                continue

            # including j
            data['tem'].append(tcons[j])  #file.variables['TEMP_ADJUSTED'][i][j]) 
            data['sal'].append(abssal[j])
            data['pre'].append(-file.variables['PRES_ADJUSTED'][i][j])
            data['den'].append(sigma0[j])
            data['nsq'].append(nsquared[j])
            if 'DOXY_ADJUSTED' in file.variables.keys():
                data['oxy'].append(file.variables['DOXY_ADJUSTED'][i][j])
            elif 'DOX2_ADJUSTED' in file.variables.keys():
                data['oxy'].append(file.variables['DOX2_ADJUSTED'][i][j])
            else:
                data['oxy'].append(numpy.nan)

            # not including j
            data['lon'].append(file.variables['LONGITUDE'][i])
            data['lat'].append(file.variables['LATITUDE'][i])
            data['dates'].append(normdates[i])
            data['cyk'].append(i)
            data['tim'].append(normdates[i]) # this is just here for legacy compatibility reasons for my older scripts
            data['timestamps'].append(normdates[i].timestamp())
            #data['bvf'] = -9.81/data['den']*numpy.gradient(data['den'])/numpy.gradient(data['pre'])

    print(len(data['tem']), len(data['den']), len(data['lon']))
    dfsoc = pandas.DataFrame.from_dict(data)
    # dfsoc.to_pickle(filepath)
    return dfsoc


def create_datagrid(df, variable, xioverwrite=None, xres=500):
    """ This function is usefull to grid/regrid two datasets
    (e.g. CTD trajectories ) over a common time period (difference plots)

    Parameters:
        df (pandas.DataFrame): Float data frame
        variable (str): variable to regrid and return
        xioverwrite (timeaxis): Only specifiy  to limit dataset to same
        time axis as different previously specified gridded data

    Returns:
        xi: Time axis
        yi: depth axis
        grid_z0: numpy.array of regridded data"""
    yi = numpy.linspace(0, -2000, 401)
    if xioverwrite is not None:
        xi = xioverwrite
    else:
        # breakpoint()
        xi = numpy.linspace(min(df['timestamps'].astype('int')),
            max(df['timestamps'].astype('int')), xres)
    if variable == 'nsq':
        xi, yi, grid_z0den = create_datagrid(df=df, variable='den', xioverwrite=xioverwrite)
        xi, yi, grid_z0pre = create_datagrid(df=df, variable='pre', xioverwrite=xioverwrite)
        grid_z0 = -9.81/grid_z0den*numpy.gradient(grid_z0den, axis=0)/numpy.gradient(grid_z0pre, axis=0)
        with warnings.catch_warnings():
            # this will suppress all warnings in this block
            warnings.simplefilter("ignore")
            grid_z0[grid_z0<0] = 0
            grid_z0[grid_z0>0.006] = 0.006 # the areas are indicated with the extend = 'both' bars      
        grid_z0[0:3,:] = 0
    else:
        grid_z0 = griddata(
            points=numpy.array(
                [numpy.array(df['timestamps']), numpy.array(df['pre'])]).T,
            values=df[variable],
            xi=(xi[None, :], yi[:, None]),
            method='linear',
            rescale=True)
    return xi, yi, grid_z0



def create_contourplot(xi, yi, grid_z0, ax, maxdepth, cmap,
    steplength=0.1, normalized=False, vmin=None, vmax=None, 
    locator=None, colors='k', contourf=True, contour=True,
    mindepth=0, linewidths=0.5, styletimeaxis=True):

    """create contourplots of already gridded data

    Parameters:
        xi: Time axis
        yi: depth axis
        grid_z0: gridded array
        ax: axis object as part of a python-subplot 
        maxdepth: y-axis lower limit (specify positive)
        cnmap: matplotlib.colormap
        steplenght: distance of contourline and level steps
        normalized: set to True for difference plots, centeres colorbars at 0
        vmin: minum variable value to plot (rest is white)
        vmax: maximum variable value to plot (rest is white)
        locator: This is only specified for contours on log scale (BVF)

    Returns:
        colors: to create colorbars, 
        lines: to create also the contour-line levels in the colorbars"""

    if not locator:
        if (vmin and vmax):
            levels=numpy.arange(vmin, vmax+steplength, steplength)
        else:
            if normalized:
                symmax = numpy.nanmax(abs(grid_z0)) # to center sym colorscale
                levels=numpy.arange(-symmax, symmax+steplength, steplength)
            else:
                levels=numpy.arange(numpy.nanmin(grid_z0), 
                    numpy.nanmax(grid_z0)+steplength,steplength)
    else:
        levels=None

    xi = [datetime.datetime.fromtimestamp(fdate) for fdate in xi]
    if contour:
        lines = ax.contour(xi,yi,grid_z0,levels=levels,linewidths=linewidths, 
            vmin=vmin, vmax=vmax, locator=locator, colors=colors, alpha=0.5,linestyle='-')
    if contourf:
        colors = ax.contourf(xi,yi,grid_z0,levels=levels,cmap=cmap,vmin=vmin, 
            vmax=vmax, locator=locator, extend='both')

    if styletimeaxis:
        style_timeaxis(ax)

    ax.set_ylim(-maxdepth,-mindepth)

    #years = mdates.YearLocator()   # every year
    #months = mdates.MonthLocator()  # every month
    #years_fmt = mdates.DateFormatter('%Y')
    #months_fmt = mdates.DateFormatter('%m')

    #ax.set_ylim(-maxdepth,0)
    #ax.xaxis.set_minor_locator(months)
    #ax.xaxis.set_minor_formatter(months_fmt) 
    #ax.xaxis.set_major_locator(years)
    #ax.xaxis.set_major_formatter(years_fmt)

    #ax.xaxis.grid(which='both', color='lightgrey')
    #ax.tick_params(axis='x', which='minor', colors='grey')
    if contour and contourf:
        return colors, lines
    elif contour and not contourf:
        return lines
    else:
        return colors


def style_timeaxis(ax, grid=True, color='grey', minor=True, monthnames=False):
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    years_fmt = mdates.DateFormatter('%Y')
    if minor:
        formatter = '%B' if monthnames else '%m'
        months_fmt = mdates.DateFormatter(formatter)
    else:
        months_fmt = mdates.DateFormatter('')
    ax.xaxis.set_minor_locator(months)
    ax.xaxis.set_minor_formatter(months_fmt) 
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)

    if grid:
        ax.xaxis.grid(which='minor', color='lightgrey')
        ax.xaxis.grid(which='major', color='darkgrey')
    ax.tick_params(axis='x', which='minor', colors=color)
    ax.tick_params(axis='x', which='major', colors='k')


def create_mld(df, oneprofile=False, maxbvf=False, xres=500):
    """ calculate the mld-depth, based on a density difference criteria """
    # Under the sea ice, float samples does all the way to the surface,
    # to keep criteria consistent we form difference from water properties at 20m depth at all times,
    mld = numpy.array([])

    if oneprofile:
        df = df.drop_duplicates(subset = ["sal"])
        df = df.drop_duplicates(subset = ["den"])
        df = df.drop_duplicates(subset = ["pre"])
        newx = numpy.arange(20,200,1)
        newy = numpy.interp(newx, -df['pre'].values, df['den'].values)
        depth20 = numpy.argmin(abs(newx-20)) # index on depth axis (yi) which is closest to 20 m
        diff = newy-(newy[depth20]+mld_density_diff)
        diff[diff<=0] = numpy.nan
        if numpy.isnan(diff).all():
            mld_onetime=numpy.nan
        else:
            mld_onetime = newx[numpy.nanargmin(diff)]
        
        # old solution
        """
        depth20 = numpy.argmin(abs(df['pre']+20)) # index on depth axis (yi) which is closest to 20 m
        profile = profile[profile['pre']<=-20]
        diff = df['den']-(df['den'].iloc[depth20]+mld_density_diff)
        diff[diff<=0] = numpy.nan
        mld_onetime = df['pre'].iloc[numpy.argmin(diff)]
        """
        return -mld_onetime


    if type(df) == pandas.core.frame.DataFrame:
        # long list of values instead of single profiles
        xi, yi, grid_z0 = create_datagrid(df, 'den', xioverwrite=None, xres=xres)        
        depth20 = numpy.argmin(abs(yi+20)) # index on depth axis (yi) which is closest to 20 m
        for i in range(0,len(grid_z0[0,])):
            try:
                mld_onetime = yi[
                numpy.nanargmin(
                    abs(
                        grid_z0[:,i]-
                            (grid_z0[depth20:,i][numpy.isfinite(grid_z0[depth20:,i])][0]+mld_density_diff)))]
            except:
                mld_onetime = numpy.nan 
                print('Warning: skipped one value in MLD computation')
            mld = numpy.append(mld, mld_onetime)

    elif type(df) == xarray.core.dataset.Dataset:
        for i in range(0,len(df['TEMP'])):
            if (numpy.isnan(df['PRES_ADJUSTED'][i]).all()):# or numpy.isnan(df['PSAL_ADJUSTED'][i]).all() or numpy.isnan(df['TEMP_ADJUSTED'][i]).all()):
                mld_onetime = numpy.nan
            else:
                depth20 = numpy.nanargmin(abs(numpy.array(df['PRES_ADJUSTED'][i])-20)) # index on depth axis which is closest to 20 m
                tcons = gsw.CT_from_t(df['PSAL_ADJUSTED'][i], df['TEMP_ADJUSTED'][i], df['PRES_ADJUSTED'][i])
                sigma0 = gsw.density.sigma0(df['PSAL_ADJUSTED'][i], tcons)
                argument = abs(sigma0 - (sigma0[depth20]+mld_density_diff))
                if numpy.isnan(argument).all():
                    mld_onetime = numpy.nan
                else:
                    mld_onetime = df['PRES_ADJUSTED'][i][numpy.nanargmin(argument)]
            mld = numpy.append(mld, mld_onetime)


    else:
        for profile in df:
            profile = profile.drop_duplicates(subset = ["sal"])
            profile = profile.drop_duplicates(subset = ["den"])
            profile = profile.drop_duplicates(subset = ["pre"])
            profile = profile.reset_index()
            if len(profile)<=1:
                mld_onetime=numpy.nan
            else:
                if maxbvf:

                    if numpy.isnan(profile['nsq']).all():
                        mld_onetime=0
                    else:
                        #print('using maxbvf method')
                        #breakpoint()
                        #profile = profile[abs(profile['pre'])>20]
                        #try:
                        #    mld_onetime = -profile['pre'].iloc[numpy.nanargmax(profile['nsq'])]
                        #except:
                        #    breakpoint()
                        newx = numpy.arange(20,200,1)
                        newy = numpy.interp(newx, -profile['pre'].values, profile['nsq'].values)
                        depth20 = numpy.argmin(abs(newx-20)) # index on depth axis (yi) which is closest to 20 m
                        mld_onetime = newx[numpy.nanargmax(newy)]
                        #print('pass', mld_onetime)
                        #breakpoint()
                        #mld_onetime = -profile['pre'].iloc[numpy.nanargmax(profile['nsq'])]
                        #diff = newy-(newy[depth20]+mld_density_diff)
                        #diff[diff<=0] = numpy.nan
                        #try:
                        #    mld_onetime = newx[numpy.nanargmin(diff)]
                        #except:
                        #    breakpoint()
                else:
                    # print('branch4b')
                    newx = numpy.arange(20,200,1)
                    newy = numpy.interp(newx, -profile['pre'].values, profile['den'].values)
                    depth20 = numpy.argmin(abs(newx-20)) # index on depth axis (yi) which is closest to 20 m
                    diff = newy-(newy[depth20]+mld_density_diff)
                    diff[diff<=0] = numpy.nan
                    try:
                        mld_onetime = newx[numpy.nanargmin(diff)]
                    except:
                        breakpoint()
                    # old solution
                    """
                    depth20 = numpy.argmin(abs(profile['pre']+20)) # index on depth axis (yi) which is closest to 20 m
                    diff = profile['den']-(profile['den'].iloc[depth20]+mld_density_diff)
                    diff[diff<=0] = numpy.nan
                    mld_onetime = profile['pre'].iloc[numpy.argmin(diff)]
                    """
            mld = numpy.append(mld, -mld_onetime)

    # print('The length of the mld is %s'%len(mld))
    return mld

def create_ww_lower(df, oneprofile=False):
    """ calculate the depth of the winter water as the deepest point having a 
        temperature below e.g. -0.5°C """ 
    ww_temp_max = -0.5
    ww_lower = numpy.array([])

    if oneprofile:
        winterwater = numpy.where(df['tem']<ww_temp_max)
        if len(winterwater[0]):
            #breakpoint()
            #try:
            depth = df['pre'].iloc[[numpy.nanmax(numpy.where(df['tem']<ww_temp_max))]].values
            depth = depth[0]
            #except:
            #    breakpoint()
        else:
            depth = numpy.nan
        #print(type(depth))
        return depth

    if type(df) == pandas.core.frame.DataFrame:
        # long list of values instead of single profiles
        xi, yi, grid_z0 = create_datagrid(df, 'tem', xioverwrite=None)
        for i in range(0,len(grid_z0[0,])):
            winterwater = numpy.where(grid_z0[:,i]<ww_temp_max)
            if len(winterwater[0]):
            #try:
                depth = yi[numpy.nanmax(numpy.where(grid_z0[:,i]<ww_temp_max))]
            else:
                depth = numpy.nan
            #except:
            #    breakpoint()
            ww_lower = numpy.append(ww_lower, depth)
    return ww_lower

def create_isopycnal(df, variable, density_value=None, depth_value=None, oneprofile=False):
    """ calculate the depth of an isopycnal of value density_value 
    input can be a df(old) that is then converted to a datagrid that 
    includes the densities 
    Input:
    variable could be den, rho, gamman...

    Returns:
    isopycnal (numpy.array): depth of the isopycnal on grid xi
    spicyness (numpy.array): spicyness along that isopycnal on grid xi""" 
    if (density_value and depth_value) or (not density_value and not depth_value):
    	raise Exception("You must either specifiy depth or density, not both") 
    isopycnal = numpy.array([])
    spiciness = numpy.array([])

    if oneprofile:
        # this one profile implementation is 
        # untested and might not work
        winterwater = numpy.where(df[variable]<density_value)
        if len(winterwater[0]):
            depth = df['pre'].iloc[[numpy.nanmin(numpy.where(df[variable]>density_value))]].values
        else:
            depth = numpy.nan
        return pycnocline_depth

    if type(df) == pandas.core.frame.DataFrame:
        # long list of values instead of single profiles
        xi, yi, grid_z0_den = create_datagrid(df, variable, xioverwrite=None)
        xi, yi, grid_z0_tem = create_datagrid(df, 'tem', xioverwrite=None)
        xi, yi, grid_z0_sal = create_datagrid(df, 'sal', xioverwrite=None)
        for i in range(0,len(grid_z0_den[0,])):
            #with warnings.catch_warnings():
            #    # this will suppress all warnings in this block
            #    warnings.simplefilter("ignore")
            if density_value:
                dense_water = numpy.where(grid_z0_den[:,i]>density_value)
                # breakpoint()
                if len(dense_water[0])>0:
                    try:
                        depth_index = numpy.nanmin(numpy.where(grid_z0_den[:,i]>density_value))
                    except:
                        breakpoint()
                    #    print('skipped density value for index %s'%i)
                    spice = gsw.spiciness0(grid_z0_sal[:,i][depth_index], grid_z0_tem[:,i][depth_index])
                    depth = yi[depth_index]
                else:
                    depth = numpy.nan
                    spice = numpy.nan
                    print('error or skip: no density value:', density_value)
                    # breakpoint()
            elif depth_value:
                    # if computing the spice strictly along one depth index (not recommended and not used by me)
                    depth_index = numpy.nanargmin(abs(yi+depth_value))
                    spice = gsw.spiciness0(grid_z0_sal[:,i][depth_index], grid_z0_tem[:,i][depth_index])
                    depth = yi[depth_index]
            isopycnal = numpy.append(isopycnal, depth)
            spiciness = numpy.append(spiciness, spice)
    return isopycnal, spiciness

#def create_isopycnals_at_depths(df, variable, deptharray):
#    # 1. compute suitable density values that correlate with the chosen depths
#    xi, yi, grid_z0 = create_datagrid(df, variable, xioverwrite=None)
#    depth_indexarray= numpy.array([numpy.argmin(abs(yi+i)) for i in deptharray])
#    densities = numpy.nanmean(grid_z0, axis=1) # densities for all yi
#    densities = densities[depth_indexarray] # densities for the deptharray corr. yi


def create_spice2d(df, variable, oneprofile=False):
    """ calculate the depth of an isopycnal of value density_value 
    input can be a df(old) that is then converted to a datagrid that 
    includes the densities 

    Returns:
    spicyness (numpy.array): spicyness along that isopycnal on grid xi""" 

    spiciness2d = []
    depths2d = []
    densities2d = []
    nsq2d = []

    xi, yi, grid_z0_den = create_datagrid(df, variable, xioverwrite=None)
    xi, yi, grid_z0_tem = create_datagrid(df, 'tem', xioverwrite=None)
    xi, yi, grid_z0_sal = create_datagrid(df, 'sal', xioverwrite=None)
    xi, yi, grid_z0_pre = create_datagrid(df, 'pre', xioverwrite=None)
    xi, yi, grid_z0_nsq = create_datagrid(df, 'nsq', xioverwrite=None)
    if variable=='rho':
        # This is the case when the function is used with in situ density value
        dens_ax = numpy.linspace(1027, 1038, 500)#176)
    elif variable=='den':
        # This is the case if using potential density sigma0
        dens_ax = numpy.linspace(27.1, 27.9, 500)#176)
    else:
        # This is the case if using gamman
        dens_ax = numpy.linspace(27.4, 28.4)
    #breakpoint()
    for density_value in dens_ax:
        spiciness = numpy.array([])
        depths = numpy.array([])
        densities = numpy.array([])
        nsqs = numpy.array([])
        for i in range(0,len(grid_z0_den[0,])):
            try:
                depth20 = numpy.nanargmin(abs(grid_z0_pre[:,i]+20))
            except:
                depth = numpy.nan
                spice = numpy.nan
                density = numpy.nan
                nsq = numpy.nan
            if (len(numpy.where(grid_z0_den[depth20:,i]>density_value)[0])>0 and
                len(numpy.where(grid_z0_den[depth20:,i]<density_value)[0])>0):
                depth_index = numpy.nanmin(numpy.where(grid_z0_den[:,i]>density_value))
                spice = gsw.spiciness0(grid_z0_sal[:,i][depth_index], grid_z0_tem[:,i][depth_index])
                #spice = gsw.spiciness0(grid_z0_sal[:,i][depth_index], grid_z0_tem[:,i][depth_index])
                nsq = grid_z0_nsq[:,i][depth_index]
                depth = yi[depth_index]
                density=density_value
            else:
                depth = numpy.nan
                spice = numpy.nan
                density = numpy.nan
                nsq = numpy.nan

            depths = numpy.append(depths, depth)
            spiciness = numpy.append(spiciness, spice)
            densities = numpy.append(densities,density)
            nsqs = numpy.append(nsqs, nsq)
        
        spiciness2d.append(spiciness)
        depths2d.append(depths)
        densities2d.append(densities)
        nsq2d.append(nsqs)
        
    return dens_ax, spiciness2d, depths2d, densities2d, nsq2d


def create_density_along_mld(df, variable):
    mld_density = numpy.array([])
    if type(df) == pandas.core.frame.DataFrame:
        # long list of values instead of single profiles
        xi, yi, grid_z0 = create_datagrid(df, variable, xioverwrite=None)   
        depth20 = numpy.argmin(abs(yi+20)) # index on depth axis (yi) which is closest to 20 m

        for i in range(0,len(grid_z0[0,])):
            try:
                mld_onetime = grid_z0[
                    numpy.nanargmin(
                        abs(
                            grid_z0[:,i]-
                                (grid_z0[depth20:,i][numpy.isfinite(grid_z0[depth20:,i])][0]+mld_density_diff))),i]
            except:
                mld_onetime = numpy.nan
                print('skipped density_along_mld for index %s'%i)
            mld_density = numpy.append(mld_density, mld_onetime)
    return mld_density

def create_density_along_ww_lower(df, variable, oneprofile=False):
    """ calculate the depth of the winter water as the deepest point having a 
        temperature below e.g. -0.5°C """ 
    ww_temp_max = -0.5
    ww_lower_density = numpy.array([])

    if type(df) == pandas.core.frame.DataFrame:
        xi, yi, grid_z0_den = create_datagrid(df, variable, xioverwrite=None)        
        # long list of values instead of single profiles
        xi, yi, grid_z0_tem = create_datagrid(df, 'tem', xioverwrite=None)
        for i in range(0,len(grid_z0_tem[0,])):
            winterwater = numpy.where(grid_z0_tem[:,i]<ww_temp_max)
            if len(winterwater[0]):
                density = grid_z0_den[numpy.nanmax(numpy.where(grid_z0_tem[:,i]<ww_temp_max)),i]
            else:
                density = numpy.nan
            ww_lower_density = numpy.append(ww_lower_density, density)
    return ww_lower_density

def integrate(df, variable, from_depth, to_depth, oneprofile=False, xres=500, returnmean=False):
    result = numpy.array([])
    if oneprofile:
        df = df.drop_duplicates(subset = ["sal"])
        df = df.drop_duplicates(subset = ["den"])
        df = df.drop_duplicates(subset = ["pre"])
        newpre = numpy.linspace(-to_depth,-from_depth, 100)
        newsal = numpy.interp(newpre, -df['pre'].values, df['sal'].values)
        newden = numpy.interp(newpre, -df['pre'].values, df['den'].values)
        newtem = numpy.interp(newpre, -df['pre'].values, df['tem'].values)
        newnsq = numpy.interp(newpre, -df['pre'].values, df['nsq'].values)

        df2 = pandas.DataFrame(dict(pre=newpre, sal=newsal, den=newden, tem=newtem, nsq=newnsq))
        #breakpoint()
        if variable=='tem':
            cp   = gsw.cp_t_exact(df2['sal'],df2['tem'],df2['pre'])
            # This is not quite the absolute heat amount, because I use sigma0 as density
            # to change to absolute, it might be enough to add 1000kg/m³?
            m_g  = numpy.gradient(-df2['pre'])*(df2['den']+1000) # mass per vertical grid cell * dens = weight
            heat = cp*m_g*(df2['tem']+273.15) # [J/kg/K]*[kg]*[K] # per gird cell
            columncontent = numpy.nansum(heat)
            tempmean = numpy.nanmean(df2['tem'])
            if returnmean:
                return columncontent, tempmean
            else:
                return columncontent
        if variable=='sal':
            m_g  = numpy.gradient(df2['pre']
                                 )*(df2['den']+1000) # mass per vertical grid cell * dens = weight
            salt = m_g*(df2['sal']/1000) # kg*[g/kg]
            columncontent = numpy.nansum(salt)
            saltmean = numpy.nanmean(df2['sal'])
            if returnmean:
                return columncontent, saltmean
            else:
                return columncontent
        if variable=='nsq':
            columncontent = numpy.nansum(df2['nsq'])
            return columncontent

        """ old solution
        from_depth_index = numpy.argmin(abs(df['pre']-from_depth))
        to_depth_index = numpy.argmin(abs(df['pre']-to_depth))
        print('I should implement interpolation')
        if variable=='sal':
            m_g  = numpy.gradient(-df['pre'][to_depth_index:from_depth_index]
                                 )*(df['den'][to_depth_index:from_depth_index]+1000) # mass per vertical grid cell * dens = weight
            salt = m_g*(df['sal'][to_depth_index:from_depth_index]/1000) # kg*[g/kg]
            columncontent = numpy.nansum(salt) 
            return columncontent
        end old solution """

    if type(df) == pandas.core.frame.DataFrame:
        xi, yi, grid_z0_tem = create_datagrid(df, 'tem', xioverwrite=None, xres=xres)
        xi, yi, grid_z0_sal = create_datagrid(df, 'sal', xioverwrite=None, xres=xres)
        xi, yi, grid_z0_pre = create_datagrid(df, 'pre', xioverwrite=None, xres=xres)
        xi, yi, grid_z0_den = create_datagrid(df, 'den', xioverwrite=None, xres=xres)
        xi, yi, grid_z0_nsq = create_datagrid(df, 'nsq', xioverwrite=None, xres=xres)

        from_depth_index = numpy.argmin(abs(yi-from_depth))
        to_depth_index = numpy.argmin(abs(yi-to_depth))

        for i in range(0,len(grid_z0_den[0,])):
            if variable=='tem':
                # I turn sigma0 into potential density by adding 1000m that should be legit
                # with t the in situ temperature! Need to change this if publishing results!
                cp   = gsw.cp_t_exact(grid_z0_sal[to_depth_index:from_depth_index, i],
                                      grid_z0_tem[to_depth_index:from_depth_index, i],
                                      -grid_z0_pre[to_depth_index:from_depth_index, i]) #[J/kg/K]
                # This is not quite the absolute heat amount, because I use sigma0 as density
                # to change to absolute, it might be enough to add 1000kg/m³?
                m_g  = numpy.gradient(-grid_z0_pre[to_depth_index:from_depth_index, i]
                                    )*(grid_z0_den[to_depth_index:from_depth_index, i]+1000) # mass per vertical grid cell * dens = weight
                heat = cp*m_g*(grid_z0_tem[to_depth_index:from_depth_index,i]+273.15) # [J/kg/K]*[kg]*[K] # per gird cell
                columncontent = numpy.nansum(heat)
            if variable=='sal':
                m_g  = numpy.gradient(-grid_z0_pre[to_depth_index:from_depth_index, i]
                                     )*(grid_z0_den[to_depth_index:from_depth_index, i]+1000) # mass per vertical grid cell * dens = weight
                salt = m_g*(grid_z0_sal[to_depth_index:from_depth_index, i]/1000) # kg*[g/kg]
                columncontent = numpy.nansum(salt) 

            if variable=='nsq':
                columncontent = numpy.nansum(grid_z0_nsq[to_depth_index:from_depth_index, i])

            result = numpy.append(result, columncontent)

    return result


def smooth_profile(profile, smooth_params=dict(sal=0.00005, tem=0.0025, den=0.00004),
                   upperdepthlimit=200, lowerdepthlimit=1600, skipnan=False):
    rms = []
    #indices = []
    #dates = []
    #newprofile = {}
    profile_int = {}
    profile = profile.drop_duplicates(subset = ["sal"])
    profile = profile.drop_duplicates(subset = ["den"])
    if numpy.nanmin(profile['pre']>-lowerdepthlimit): #or (i==0):
        # skip the profiles that were sampled to 500 m depth only
        if skipnan:
            return []
        else:
            rms.append(numpy.nan)

    else:
        # really good value for all (except maybe direct visualisation of splines) are
        # sal -> s=0.00005; tem -> s=0.00015; den -> s=0.00004
        profile_int['sal'] = scipy.interpolate.UnivariateSpline(
            -profile['pre'], profile['sal'],s=smooth_params['sal'])#s=0.00001)
        profile_int['tem'] = scipy.interpolate.UnivariateSpline(
            -profile['pre'], profile['tem'],s=smooth_params['tem'])#s=0.00005)
        profile_int['den'] = scipy.interpolate.UnivariateSpline(
            -profile['pre'], profile['den'],s=smooth_params['den'])#s=0.00005) # original 0.00004

        profile_int['sal'] = profile_int['sal'](
            numpy.arange(upperdepthlimit,lowerdepthlimit,1))
        profile_int['tem'] = profile_int['tem'](
            numpy.arange(upperdepthlimit,lowerdepthlimit,1))
        profile_int['den2'] = gsw.sigma0(profile_int['sal'], profile_int['tem'])

        profile_int['den'] = profile_int['den'](
            numpy.arange(upperdepthlimit,lowerdepthlimit,1))

        profile_int['pre'] = numpy.arange(upperdepthlimit,lowerdepthlimit,1)
        profile_int['spice'] = gsw.spiciness0(profile_int['sal'], profile_int['tem'])

    return profile_int


def compute_diapycnal_spice_variations(profiles, upperdepthlimit=200, 
                                       lowerdepthlimit=1000, 
                                       skipnan=False, SOCCOM=False, original_method=False):
    rms = []
    indices = []
    dates = []
    for i,profile in enumerate(profiles):

        # Explanation of this subsampling part:
        # First of all, it could be left out and the results would still be valid, 
        # however, they would not be intercomparable between different floats.
        # The ARGO/SOCCOM data is sampled in 2m intervals to a depth of 1000m,
        # while our data is sampled in 25m intervals there. The ARGO/SOCCOM data
        # is sampled in 100m intervals below 1000m, while our data is sampled with
        # 25m there. We subsample the foreign data to make the rms analysis comparable
        
        if SOCCOM:
            #subsampling filter is active for the higher sampled argo/soccom profiles
            #print('subsampling is active')
            profile_upper1000 = profile[profile.pre>-1000].iloc[0::12]
            profile_lower1000 = profile[profile.pre<-1000]
            profile = pandas.concat([profile_upper1000, profile_lower1000], axis=0)

        profile_int = smooth_profile(profile, upperdepthlimit=upperdepthlimit, 
                                     lowerdepthlimit=lowerdepthlimit, skipnan=True)
        if not len(profile_int):
            continue

        #if original_method:
        if i in [0,1 ,258,259]:
            # first profiles where noisy (airbubbles?)
            # 258,259 have artifacts
            continue
        dates.append(profile.dates.iloc[0])

        profile2 = copy.deepcopy(profile)
        profile2 = profile2[profile2.pre>-lowerdepthlimit]
        
        profile2['dep'] = 500
        profile2['den_dep'] = gsw.rho(profile2['sal'], profile2['tem'], profile2['dep'])

        variables = ['sal', 'den', 'pre']
        for variable in variables:
            profile2 = profile2.drop_duplicates(subset = [variable])

        
        ip = {} #dictionary to take up interpolated profiles
        dz = 25
        ip['pres'] = numpy.arange(upperdepthlimit,lowerdepthlimit+dz,dz)
        for variable in ['sal', 'tem', 'den_dep', 'pre', 'nsq']:
            ip[variable] = numpy.interp(ip['pres'],#newx, 
                           -profile2['pre'].values, 
                            profile2[variable].values)

        ip = pandas.DataFrame.from_dict(ip)
        ip = ip.rolling(3, center=True).mean()

        rho = gsw.rho(ip['sal'],
                      ip['tem'],
                      ip['pre'])
        alpha = gsw.alpha(ip['sal'],
                          ip['tem'],
                          ip['pre'])
        beta = gsw.beta(ip['sal'],
                        ip['tem'],
                        ip['pre'])

        def diff1(ips):
            diff1 = numpy.zeros(len(ips))
            for i in range(2,len(ips)-2):
                diff1[i] = (-ips[i+2]+8*ips[i+1]-8*ips[i-1]+ips[i-2])/(12*dz)
            return diff1

        def diff2(ips):
            diff2 = numpy.zeros(len(ips))
            for i in range(2,len(ips)-2):
                diff2[i] = (-ips[i+2]+16*ips[i+1]-30*ips[i]+16*ips[i-1]-ips[i-2])/(12*dz**2)
            return diff2

        t_ss = 2*rho**2*alpha*beta*(diff1(ip['sal'])*diff2(ip['tem'])-diff1(ip['tem'])*diff2(ip['sal']))/diff1(ip['den_dep'])**3
        rms = numpy.append(rms, numpy.sqrt(numpy.nanmean(t_ss**2)))          
        indices.append(i)

    return rms, indices, dates



def geocalc(lat1, lon1, lat2, lon2):
    """ calculates the distance between a pair of coordinates, 
    used to calculate the distance between to drifting floats at the same time """
    lat1 = numpy.radians(float(lat1))
    lon1 = numpy.radians(float(lon1))
    lat2 = numpy.radians(float(lat2))
    lon2 = numpy.radians(float(lon2))

    dlon = lon1 - lon2
    EARTH_R = 6372.8

    y = numpy.sqrt(
        (numpy.cos(lat2) * numpy.sin(dlon)) ** 2
        + (numpy.cos(lat1) * numpy.sin(lat2) - numpy.sin(lat1) * numpy.cos(lat2) * numpy.cos(dlon)) ** 2
        )
    x = numpy.sin(lat1) * numpy.sin(lat2) + numpy.cos(lat1) * numpy.cos(lat2) * numpy.cos(dlon)
    c = math.atan2(y, x)
    distance = EARTH_R * c
    return distance

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))


def profile_colors(profiles, df_col, condition):
    # condition can be: slope, depth, longitude, latitude
    # define facecolors of the profiles or scatters
    results = []
    df_col['slope'] = numpy.sqrt(df_col['depth_topo_gradx']**2+df_col['depth_topo_grady']**2)
    norm = normdict[condition]
    for pindex,profile in enumerate(profiles):
        resultsdict = dict(index=pindex, color=None, zorder=None)
        #resultsdict['index']=pindex
        #value = df_col.sel(expver=5).isel(time=10)[condition]
        try:
            value = df_col.sel(time=profile.iloc[0].dates.to_pydatetime())[condition]
        except:
            # print('missing out one profile')
            resultsdict['color']='lightgrey'#colors.append('lightgrey')
            resultsdict['zorder']=-1#zorders.append(-1)
            continue
        rgba = cmapdict[condition](normdict[condition](value))
        resultsdict['color']=rgba#colors.append(rgba)
        #indices.append(index)
        resultsdict['zorder']=1#zorders.append(1)
        #print(resultsdict)
        results.append(resultsdict)
    return pandas.DataFrame(results).set_index('index')#results#colors, zorders, indices


def profile_threshold_colors(profiles, df_col, condition, key, original_method=False):
    # define edgecolors, e.g. for rms mixing profiles
    
    # differentiate between SOCCOM and our floats, to equalize vertical resolution later on in 
    # 'compute_diapycnal_spice_variations'
    if key in ['300234067208900.nc', '300234068638900.nc']:
        SOCCOM=False #rmslimit=5000
    else:
        SOCCOM=True#rmslimit=100
    #if orginal_method:
    rmslimit=2500
    #else:
    #    rmslimit=5000
    results = []
    df_col['slope'] = numpy.sqrt(df_col['depth_topo_gradx']**2+df_col['depth_topo_grady']**2)
    rmsdeep, indices, dates = compute_diapycnal_spice_variations(
                                   profiles, 
                                   upperdepthlimit=500, 
                                   lowerdepthlimit=1550, 
                                   SOCCOM=SOCCOM,
                                   original_method=False)
    rms, indices, dates = compute_diapycnal_spice_variations(
                                   profiles, 
                                   upperdepthlimit=200, 
                                   lowerdepthlimit=1000, 
                                   SOCCOM=SOCCOM,
                                   original_method=False)
    # breakpoint()
    dictionary = {'rms': rms, 'indices': indices, 'dates': dates}
    df = pandas.DataFrame(data=dictionary)
    df = df.set_index('indices')
   
    for pindex, profile in enumerate(profiles):
        resultsdict = dict(index=pindex, color=None, zorder=None)
        #print(pindex, indices)
        if not pindex in indices:
            # profile with too shallow maximum depth
            continue 
        try:
            depth = df_col.sel(time=profile.iloc[0].dates.to_pydatetime()).depth_topo
            topoy = df_col.sel(time=profile.iloc[0].dates.to_pydatetime()).depth_topo_grady
            topox = df_col.sel(time=profile.iloc[0].dates.to_pydatetime()).depth_topo_gradx
            latitude = df_col.sel(time=profile.iloc[0].dates.to_pydatetime()).latitude
            longitude = df_col.sel(time=profile.iloc[0].dates.to_pydatetime()).longitude
        except:
            results.append(resultsdict)
            continue
            
        #breakpoint()
        # shallow (0-500m profile - continue)
        if max(abs(profile['pre'])) < 700:
            results.append(resultsdict)
            continue 
            
        if condition in ['longitude', 'latitude', 'depth_topo', 'slope']:
            try:
                value = df_col.sel(time=profile.iloc[0].dates.to_pydatetime())[condition]
            except:
                # print('missing out one profile')
                resultsdict['color']='lightgrey'#colors.append('lightgrey')
                resultsdict['zorder']=-1#zorders.append(-1)
                continue
            rgba = cmapdict[condition](normdict[condition](value))
            resultsdict['color']=rgba#colors.append(rgba)
            #indices.append(index)
            resultsdict['zorder']=1#zorders.append(1)
            #print(resultsdict)
            #results.append(resultsdict)
        # breakpoint()
        if condition == 'rms':
            if pindex == 0:
                # first profile is often noisy/wrong
                results.append(resultsdict)
                continue
            try:
                if df.loc[pindex]['rms']>rmslimit:#rms[pindex]>rmslimit:
                    # print('decided for red color due to rms=%s'%rms[pindex])
                    resultsdict['color']='red'
                    resultsdict['zorder']=4
                else:
                    resultsdict['color']=None
                    resultsdict['zorder']=3
            except:
                breakpoint()
                #print('skipped one profile due to undefined rms')
                
        if condition == 'floatkey':
            resultsdict['color']=filenames[key]#[filenames[key]*len(df_col)]
            
        if condition == 'region':
            boxes = [dict(x=2.05, y=-64.7, w=5, h=2, color='black', zorder=2),
                     dict(x=-2.5, y=-67, w=4.5, h=4, color='tab:green', zorder=3),
                     dict(x=2.05, y=-67, w=5, h=2.25, color='orange', zorder=2)]
            #for element in df_col:
            #    breakpoint()
            #    latitude = element.latitude
            #    longitude = element.longitude
            resultsdict['color'] = 'lightgrey' # this is the scatter outside of all boxes
            for box in boxes:
                if ((longitude>box['x']) and (longitude<box['x']+box['w']) and
                    (latitude>box['y']) and (latitude<box['y']+box['h'])):
                     resultsdict['color']=box['color']
                     resultsdict['zorder']=box['zorder']
                if not resultsdict['color']:
                    resultsdict['color']='lightgrey'
        results.append(resultsdict)
    results = pandas.DataFrame(results).set_index('index')
    # breakpoint()
    return results#results#sdict#colors, zorders


def plot_colored_figure(years, months, filenames, variables, mode, colorby, edgecolorby, comparedepth, axs, original_method=False):
    # mode should by either scatter or profile 
    # years = year#numpy.arange(2018,2020,1)#[2015,2016,2017,2018,2019,2020,2021]#,2017,2018,2019]#2018,2019,2020]
    # months = [1,2,3,4,5,6,7,8,9,10,11,12]

    variables_dictionary = dict(tem='Conservative \ntemperature [°C]',
                           sal='Absolute \nsalinity [g/kg]',
                           den='Potential \ndensity sigma0 [kg/m³]',
                           gamman='Neutral \ndensity [kg/m³]',
                           spice='Spice \n[kg/m³]',
                           nsq='N² [s⁻²]')
    units_dictionary = dict(tem='[°C]',
                            sal='[g/kg]',
                            den='[kg/m³]',
                            gamman='[kg/m³]',
                            spice='[kg/m³]',
                            nsq='[s⁻²]')

    for key in filenames.keys():
        df, profiles = create_dataframe(pfloat=key, force_reconstruction=False)
        df_col = xarray.open_dataset('../data/collocation_%s'%key)
        title=key
        results = profile_threshold_colors(profiles, df_col, colorby, key, original_method=original_method)#'latitude')
        threshold_results = profile_threshold_colors(profiles, df_col, edgecolorby, key, original_method=original_method)
        for index, variable in enumerate(variables):#['tem', 'sal', 'den', 'spice']):
            counter = 0
            #breakpoint()
            #for pindex, profile in enumerate(profiles):
            #breakpoint()
            for pindex in results.index:
                profile = profiles[pindex]
                
                if pindex == 0:
                    # first profile is often noisy/wrong
                    continue
                if len(profile)==0:
                    continue
                profile['spice'] = gsw.spiciness0(profile['sal'], profile['tem'])
                if (profile.iloc[0].dates.year in years) and (
                    profile.iloc[0].dates.month in months):
                    try:
                        depth = df_col.sel(time=profile.iloc[0].dates.to_pydatetime()).depth_topo
                    except:
                        continue

                    if mode == 'profile':
                        axs[index].plot(profile[variable], profile['pre'], 
                                        color=results.loc[pindex]['color'], alpha=0.5, 
                                        zorder=results.loc[pindex]['zorder'])

                    else: # (mode == 'scatter')
                        comparedepth = comparedepth#350
                        where = 'at %sm depth'%comparedepth
                        if numpy.nanmin(profile['pre']+comparedepth) > 20:
                            # This profile has new value that is sufficiently close to 400m depth
                            print('sorted out profile in %s'%key)
                            continue
                        depthindex = numpy.nanargmin(abs(profile['pre']+comparedepth)) # depth index closest to the summand
                        #try:
                        edgecolor=threshold_results.loc[pindex]['color']
                        #except:
                        #    breakpoint()
                        if results.loc[pindex]['color'] == 'lightgrey':
                            continue
                        #if results.loc[pindex]['color'] == None:
                        #    print('skipped')
                        #    continue
                        if edgecolor=='red':
                            marker='*'
                            s=100 # markersize
                            zorder=100
                        else:
                            marker='o'
                            s=50 #markersize
                            zorder=50
                        # print(results)
                        if variable == 'nsq':
                            # Fill in NaN's...
                            mask = numpy.isnan(profile[variable].values)
                            profile[variable].values[mask] = numpy.interp(numpy.flatnonzero(mask), numpy.flatnonzero(~mask), profile[variable].values[~mask])
                        axs[index].scatter(profile[variable].values[depthindex], depth, s=s,
                                           color=results.loc[pindex]['color'], zorder=zorder,#results.loc[pindex]['zorder'],
                                           edgecolor=threshold_results.loc[pindex]['color'], marker=marker)
                        counter += 1

                            # breakpoint()
                        # print(profile[variable].values[depthindex], depth)
            print('for the variable %s, %s values were plotted'%(variable, counter))
        """
        if mode == 'scatter':
            axs[index].set_xlabel(units_dictionary[variable])
            axs[index].set_title(variables_dictionary[variable]+'\n'+where)
        """

        #axs[index].grid(zorder=100)
        axs[0].set_ylabel('bathymetry depth at float location')
            #else:
                # This is in case the profile is not in the date range specified first
            #    axs[index].plot(profile[variable], profile['pre'], color='lightgrey', alpha=0.5, zorder=-2)

def colormap_alpha(cmap):
    from matplotlib.colors import ListedColormap
    my_cmap = cmap(numpy.arange(cmap.N))
    my_cmap[:,-1] = numpy.linspace(0, 1, cmap.N) # inactive
    
    #my_cmap[:,-1] = 1
    my_cmap[0,-1] = 0
    #my_cmap[,-1] = 1

    my_cmap = ListedColormap(my_cmap)
    return my_cmap
# This file just defines a collection of variable limits for pretty contour plot and other axes limitations
# Attention: the variables are defined globally here. 
import cmocean

class full_range(object):
    def __init__(self):
        self.mins = dict(sal=33.9, tem=-1.9, den=27, nsq=0, spice=-0.8, gamman=27.8, rho=1027)
        self.maxs = dict(sal=34.9, tem=1.7, den=28, nsq=0.005, spice=-0.1, gamman=28.2, rho=1038)
        self.mcolors = dict(sal=cmocean.cm.haline, tem=cmocean.cm.thermal, 
                       den=cmocean.cm.dense, nsq=cmocean.cm.tempo, spice=cmocean.cm.thermal,
                       gamman=cmocean.cm.dense, rho=cmocean.cm.dense)
        self.steplengths = dict(sal=0.02, tem=0.1, den=0.02, nsq=0.0001, spice=0.02, gamman=0.02, rho=0.1)
        self.titles = dict(sal='Absolute salinity', tem='Conservative temperature', den='Density', nsq='N²', 
                      spice='Spiciness', heat='Integrated heat', salt='Integrated salt', 
                      isopycnal_spice='Spice along isopycnal', nsqc='Integrated N²',
                      gamman='Neutral density', rho='In situ densiy')
        self.locators = dict(sal=None, tem=None, den=None, nsq=None, spice=None, gamman=None, rho=None)
        self.units = dict(sal='g/kg', tem='°C', den='kg/m³', nsq='s⁻¹', spice='kg/m³', gamman='kg/m³', rho='kg/m³')

class events_time(object):
    def __init__(self):
        events = dict(
            polynya2016=
              [date2num(datetime.datetime(2016,7,27)), date2num(datetime.datetime(2016,8,17))],
            polynya2017=
              [date2num(datetime.datetime(2017,9,3)),date2num(datetime.datetime(2017,12,1))]
            )

class floats(object):
    def __init__(self):
        filenames = ['5905381_Sprof.nc',
             'old',
             'new',
             '5904468_Mprof.nc', 
             '5904471_Mprof.nc', 
             'GL_PR_PF_5903616.nc', 
             'GL_PR_PF_7900640.nc', 
             'GL_PR_PF_5905382.nc',
             ]
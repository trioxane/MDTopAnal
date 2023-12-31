import numpy as np


def get_number(symbol):

    numbers = {'X' :  0,
               'H' :  1,'He':  2,
               'Li':  3,'Be':  4,'B' :  5,'C' :  6,'N' :  7,'O' :  8,'F' :  9,'Ne': 10,
               'Na': 11,'Mg': 12,'Al': 13,'Si': 14,'P' : 15,'S' : 16,'Cl': 17,'Ar': 18,
               'K' : 19,'Ca': 20,'Sc': 21,'Ti': 22,'V' : 23,'Cr': 24,'Mn': 25,'Fe': 26,'Co': 27,'Ni': 28,
               'Cu': 29,'Zn': 30,'Ga': 31,'Ge': 32,'As': 33,'Se': 34,'Br': 35,'Kr': 36,
               'Rb': 37,'Sr': 38,'Y' : 39,'Zr': 40,'Nb': 41,'Mo': 42,'Tc': 43,'Ru': 44,'Rh': 45,'Pd': 46,
               'Ag': 47,'Cd': 48,'In': 49,'Sn': 50,'Sb': 51,'Te': 52,'I' : 53,'Xe': 54,
               'Cs': 55,'Ba': 56,'La': 57,'Ce': 58,'Pr': 59,'Nd': 60,'Pm': 61,'Sm': 62,'Eu': 63,'Gd': 64,'Tb': 65,'Dy': 66,'Ho': 67,'Er':  68,'Tm': 69,'Yb': 70,'Lu': 71,
               'Hf': 72,'Ta': 73,'W' : 74,'Re': 75,'Os': 76,'Ir': 77,'Pt': 78,
               'Au': 79,'Hg': 80,'Tl': 81,'Pb': 82,'Bi': 83,'Po': 84,'At': 85,'Rn': 86,
               'Fr': 87,'Ra': 88,'Ac': 89,'Th': 90,'Pa': 91,'U' : 92,'Np': 93,'Pu': 94,'Am': 95,'Cm': 96,'Bk': 97,'Cf': 98,'Es': 99,'Fm': 100,'Md':101,'No':102,'Lr':103,
               'Rf':104,'Db':105,'Sg':106,'Bh':107,'Hs':108,'Mt':109,'Ds':110,
               'Rg':111,'Cn':112,'Nh':113,'Fl':114,'Mc':115,'Lv':116,'Ts':117,'Og': 118,
               'T' : -2,'D' : -1}
    number = numbers.get(symbol)
    if number is None:
        raise ValueError("The atomic symbol |" + symbol + "| is not recognized!")
    return number


def get_wdv_radii(numbers): #2013_Dalton_Transactions_42_8617

    radii = np.array([0.00,
                      1.20, 1.43,
                      2.12, 1.98, 1.91, 1.77, 1.66, 1.50, 1.46, 1.58,
                      2.50, 2.51, 2.25, 2.19, 1.90, 1.89, 1.82, 1.83,
                      2.73, 2.62, 2.58, 2.46, 2.42, 2.45, 2.45, 2.44, 2.40, 2.40,
                      2.38, 2.39, 2.32, 2.29, 1.88, 1.82, 1.86, 2.25,
                      3.21, 2.84, 2.75, 2.52, 2.56, 2.45, 2.44, 2.46, 2.44, 2.15,
                      2.53, 2.49, 2.43, 2.42, 2.47, 1.99, 2.04, 2.06,
                      3.48, 3.03, 2.98, 2.88, 2.92, 2.95, 2.9 , 2.90, 2.87, 2.83, 2.79, 2.87, 2.81, 2.83, 2.79, 2.80, 2.74,
                      2.63, 2.53, 2.57, 2.49, 2.48, 2.41, 2.29,
                      2.32, 2.45, 2.47, 2.60, 2.54, 2.0 , 2.0 , 2.5 ,
                      3.4 , 2.8 , 2.8 , 2.93, 2.88, 2.71, 2.82, 2.81, 2.83, 3.05, 3.4 , 3.05, 2.7 , 2.7 , 2.7 , 2.7 , 2.7 ,
                      2.7 , 2.7 , 2.7 , 2.7 , 2.7 , 2.7 , 2.7 ,
                      2.7 , 2.7 , 2.7 , 2.7 , 2.7 , 2.7 , 2.7 , 2.7 ,
                      1.20, 1.20])
    return radii[numbers]


def get_covalent_radii(numbers): #2008_Dalton_Transactions_0_2832

    radii = np.array([0.00,
                      0.31, 0.28,
                      1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58,
                      1.28, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06,
                      2.03, 1.76, 1.70, 1.60, 1.53, 1.39, 1.61, 1.52, 1.5, 1.24,
                      1.32, 1.22, 1.22, 1.20, 1.19, 1.20, 1.2 , 1.16,
                      2.20, 1.95, 1.90, 1.75, 1.64, 1.54, 1.47, 1.46, 1.42, 1.39,
                      1.45, 1.44, 1.42, 1.39, 1.39, 1.38, 1.39, 1.40,
                      2.44, 2.15, 2.07, 2.04, 2.03, 2.01, 1.99, 1.98, 1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87, 1.87,
                      1.75, 1.70, 1.62, 1.51, 1.44, 1.41, 1.36,
                      1.36, 1.32, 1.45, 1.46, 1.48, 1.40, 1.50, 1.50,
                      2.60, 2.21, 2.15, 2.06, 2.00, 1.96, 1.90, 1.87, 1.80, 1.69, 1.7 , 1.7 , 1.7 , 1.7 , 1.7 , 1.7 , 1.7 ,
                      1.57, 1.49, 1.43, 1.41, 1.34, 1.29, 1.28,
                      1.21, 1.22, 1.76, 1.74, 1.57, 1.64, 1.57, 1.57,
                      0.31, 0.31])
    return radii[numbers]


def get_symbols(numbers):

    symbols = np.array(['X',
                        'H' ,'He',
                        'Li','Be','B' ,'C' ,'N' ,'O' ,'F' ,'Ne',
                        'Na','Mg','Al','Si','P' ,'S' ,'Cl','Ar',
                        'K' ,'Ca','Sc','Ti','V' ,'Cr','Mn','Fe','Co','Ni',
                        'Cu','Zn','Ga','Ge','As','Se','Br','Kr',
                        'Rb','Sr','Y' ,'Zr','Nb','Mo','Tc','Ru','Rh','Pd',
                        'Ag','Cd','In','Sn','Sb','Te','I' ,'Xe',
                        'Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu',
                        'Hf','Ta','W' ,'Re','Os','Ir','Pt',
                        'Au','Hg','Tl','Pb','Bi','Po','At','Rn',
                        'Fr','Ra','Ac','Th','Pa','U' ,'Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr',
                        'Rf','Db','Sg','Bh','Hs','Mt','Ds',
                        'Rg','Cn','Nh','Fl','Ms','Lv','Ts','Og',
                        'T' ,'D'])
    return symbols[numbers]


def is_metal(symbol):

    metals = { 'Li' : True, 'Be' : True,
               'Na' : True, 'Mg' : True, 'Al' : True,
               'K'  : True, 'Ca' : True, 'Sc' : True, 'Ti' : True, 'V'  : True, 'Cr' : True, 'Mn' : True, 'Fe' : True, 'Co' : True, 'Ni' : True,
               'Cu' : True, 'Zn' : True, 'Ga' : True, 'Ge' : True,
               'Rb' : True, 'Sr' : True, 'Y'  : True, 'Zr' : True, 'Nb' : True, 'Mo' : True, 'Tc' : True, 'Ru' : True, 'Rh' : True, 'Pd' : True,
               'Ag' : True, 'Cd' : True, 'In' : True, 'Sn' : True, 'Sb' : True,
               'Cs' : True, 'Ba' : True, 'La' : True, 'Hf' : True, 'Ta' : True, 'W'  : True, 'Re' : True, 'Os' : True, 'Ir' : True, 'Pt' : True,
               'Au' : True, 'Hg' : True, 'Tl' : True, 'Pb' : True, 'Bi' : True, 'Po' : True,
               'Fr' : True, 'Ra' : True, 'Ac' : True, 'Rf' : True, 'Db' : True, 'Sg' : True, 'Bh' : True,
               'Ce' : True, 'Pr' : True, 'Nd' : True, 'Pm' : True, 'Sm' : True, 'Eu' : True, 'Gd' : True,
               'Tb' : True, 'Dy' : True, 'Ho' : True, 'Er' : True, 'Tm' : True, 'Yb' : True, 'Lu' : True,
               'Th' : True, 'Pa' : True, 'U'  : True, 'Np' : True, 'Pu' : True, 'Am' : True, 'Cm' : True,
               'Bk' : True, 'Cf' : True, 'Es' : True, 'Fm' : True, 'Md' : True, 'No' : True, 'Lr' : True}
    return metals.get(symbol)


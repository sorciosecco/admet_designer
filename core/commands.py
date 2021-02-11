
import os
from core import settings
from core.tools import execute

def run_tauthor(infile, outfile, wdir):
    IN_path, OUT_path = os.path.join(wdir, infile), os.path.join(wdir, outfile)
    code="%s -o %s -l 1 %s"
    parameters=(settings.exec_tau, OUT_path, IN_path)
    returncode = execute(command=code % parameters, outfile=outfile, foldir=wdir)
    return returncode

def run_blabber(infile, outfile, wdir):
    IN_path, OUT_path = os.path.join(wdir, infile), os.path.join(wdir, outfile)
    code="%s -o %s %s"
    parameters=(settings.exec_bla, OUT_path, IN_path)
    returncode = execute(command=code % parameters, outfile=outfile, foldir=wdir)
    return returncode

def run_babel(infile, outfile, wdir):
    IN_path, OUT_path = os.path.join(wdir, infile), os.path.join(wdir, outfile)
    code = "%s -isdf %s -osdf %s -c --canonical"
    parameters = (settings.exec_bab, IN_path, OUT_path)
    returncode = execute(command=code % parameters, outfile=outfile, foldir=wdir)
    return returncode

#def omega(infile, outfile, wdir):
    #IN_path = os.path.join(wdir, infile)
    #OUT_path = os.path.join(wdir, outfile)
    #code="%s -in %s -out %s -buildff mmff94s -strict false -maxconfs 1"
    #parameters = (settings.exe_omega, IN_path, OUT_path)
    #if settings.VERBOSE==2: print("flipper command:", code % parameters)
    #returncode = execute(command=code % parameters, outfile=outfile, foldir=wdir)
    #return returncode

#def enumerate_stereocenters(infile, outfile, wdir):
    #IN_path = os.path.join(wdir, infile)
    #OUT_path = os.path.join(wdir, outfile)
    #if settings.NUMENA==None:
        #code = "%s -in %s -out %s"
        #parameters = (settings.exe_flipper, IN_path, OUT_path)
    #else:
        #code="%s -in %s -out %s -maxcenters %s"
        #parameters = (settings.exe_flipper, IN_path, OUT_path, settings.NUMENA)
    #if settings.VERBOSE==2:
        #print("flipper command:", code % parameters)
    #returncode = execute(command=code % parameters, outfile=outfile, foldir=wdir)
    #return returncode

#def obabel(infile, outfile, wdir, code_n):
    #IN_path = os.path.join(wdir, infile)
    #OUT_path = os.path.join(wdir, outfile)
    #if code_n==1:
        #code="%s -isdf %s -osdf %s --gen2D"
    #elif code_n==2:
        #code = "%s -isdf %s -oinchi %s"
    #elif code_n==3:
        ##code = "%s -isdf %s -osdf %s --canonical"
        #code = "%s -isdf %s -osdf %s -c --canonical"
    #elif code_n==4:
        #code = "%s -isdf %s -osmi %s"
    #parameters = (settings.exe_babel, IN_path, OUT_path)
    #if settings.VERBOSE==2:
        #print("babel command:", code % parameters)
    #returncode = execute(command=code % parameters, outfile=outfile, foldir=wdir)
    #return returncode

#def moka(infile, outfile, wdir, code_n):
    #IN_path = os.path.join(wdir, infile)
    #OUT_path = os.path.join(wdir, outfile)
    #code="%s -o %s -t %s %s"
    #if code_n==0:
        #parameters = (settings.exe_tau, OUT_path, settings.PERCENTAGE, IN_path)
    #elif code_n==1:
        #parameters = (settings.exe_bla, OUT_path, settings.PERCENTAGE, IN_path)
    #elif code_n==2:
        #code="%s -o %s -l 1 %s"
        #parameters=(settings.exe_tau, OUT_path, IN_path)
    #elif code_n==3:
        #code="%s -o %s %s"
        #parameters=(settings.exe_bla, OUT_path, IN_path)
    #if settings.VERBOSE==2:
        #print("moka command:", code % parameters)
    #returncode = execute(command=code % parameters, outfile=outfile, foldir=wdir)
    #return returncode

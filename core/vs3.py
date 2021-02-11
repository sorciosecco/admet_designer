
import os
from core import settings
from core.tools import execute

def prepare_vsscript(SDF_file, CSV_file):
    vsscript_path = "/".join(CSV_file.split('/')[:-1]+["vsscript.vsl"])
    code = 'set FIELD_PARAM = DYNAMIC;\nset GRID_SPACING = 0.5;\nset PROTONATION = AS_IS;\nset SEARCH_PAINS = NO;\nimport SDF "%s" name by MOL_NAME;\nexport datamatrix as CSV "%s" using separator ";";\n'
    parameters = (SDF_file, CSV_file)
    script=open(vsscript_path, 'w')
    script.write(code % parameters)
    script.close()
    return vsscript_path

def check_float_values(csv):
    with open(csv, "r") as c:
        lines=c.readlines()
    rc = False
    for x in lines[-1].split(';')[1:]:
        try:
            float(x)
        except ValueError:
            rc = True
    return rc

def remove_TOPPandDDRY(csv_a):
    csv_b = csv_a.replace('.csv', '110.csv')
    ofile = open(csv_b, 'w')
    for line in open(csv_a, 'r'):
        X = str.split(line.strip(), ';')
        line2 = X[:88]+X[98:121]
        ofile.write(';'.join(line2)+'\n')
    ofile.close()
    return csv_b

def remove_PAINSandCUSTOM(CSV_file):
    ocsv = CSV_file.replace('.csv', '_.csv')
    ofile = open(ocsv, 'w')
    for line in open(CSV_file, 'r'):
        ofile.write(';'.join(str.split(line.strip(), ';')[:-2])+'\n')
    ofile.close()
    return ocsv

def run_volsurf3(iname, name, isdf, wdir, run=0):
    isdf, ocsv = os.path.join(wdir, isdf), os.path.join(wdir, "%sv.csv" % iname)
    vs_script = prepare_vsscript(SDF_file=isdf, CSV_file=ocsv)
    command = '%s %s' % (settings.exec_vs3, vs_script)
    returncode = execute(command, outfile=ocsv.split('/')[-1], foldir=wdir)
    
    ocsv2=''
    if returncode==0:
        ocsv2=remove_PAINSandCUSTOM(CSV_file=ocsv)
        has_nonfloat = check_float_values(csv=ocsv2)
        if has_nonfloat:
            returncode = 1
            vs_message=" (non-float values are present)"
        else:
            vs_message=""
            
            #if settings.REMOVE:
                #ocsv2=remove_TOPPandDDRY(csv_a=ocsv)
            #else:
                #ocsv2=ocsv
            #ocsv3=ocsv2.replace('.csv', 'n.csv')
            #ocsv3=ocsv.replace('.csv', 'n.csv')
            #ofile=open(ocsv3, 'w')
            #i=0
            ##for line in open(ocsv2, 'r'):
            #for line in open(ocsv, 'r'):
                #if i>0:
                    #l = line.split(';')
                    #l[0] = name
                    #ofile.write(';'.join(l))
                #else:
                    #ofile.write(line)
                #i+=1
            #ofile.close()
    else:
        vs_message=" (input failed)"
    return returncode, ocsv2, vs_message


import os, openbabel
from rdkit import Chem

from core import settings

def add_response_tag(isdf, response_dict, response_name):
    
    n = len([1 for line in open(isdf, "r") if line.find("$$$$") != -1])
    infile=open(isdf, "r")
    osdf=open("prefiltered.sdf", "w")
    for i in range(n):
        L=[]
        while 13:
            line=infile.readline()
            L.append(line)
            if line.strip() == "$$$$":
                break
        
        L[-1] = ">  <%s>\n%s\n\n$$$$\n" % (response_name, response_dict[L[0].strip()])
        for l in L:
            osdf.write(l)
        #L=[]
    infile.close()
    osdf.close()
    os.remove(isdf)

def convert_response(y, lt, ht):
    if lt==ht:
        if y>=lt: r=1
        else: r=0
    else:
        if y>=ht: r=1
        elif y<=lt: r=-1
        else: r=0
    return r


def ob_mol(string, moltype):
    returncode, message = 0, ""
    
    try:
        obConversion = openbabel.OBConversion()
        if moltype=="smi": obConversion.SetInAndOutFormats("smi", "smi")
        else: obConversion.SetInAndOutFormats("sdf", "smi")
        mol = openbabel.OBMol()
        obConversion.ReadString(mol, string)
    except:
        returncode, message = -1, "-"
    else:
        # This checks if non-druglike atoms are present in the structure
        A=[]
        for a in mol.GetSpacedFormula().split():
            try:
                int(a)
            except ValueError:
                if a not in ["+","-"]: A.append(a)
        returncode = sum([0 if a in ["C","H","N","O","P","S","B","F","Cl","Br","I","As","Se","Si"] else 1 for a in A])
        if returncode>0: message+="+"
        else: message+="."
        
        mol.StripSalts()# This removes smaller fragments if present.
        
        na, mw = mol.NumHvyAtoms(), mol.GetMolWt()
        outSMILES = obConversion.WriteString(mol)
        if na > settings.LOWNA and na < settings.HIGHNA:# This checks if the number of non-H atoms is within the range
            message+="."
        else:
            message+="+"
            returncode=-2
        if mw > settings.LOWMW and mw < settings.HIGHMW:# This checks if the molecular weight is within the range
            message+="."
        else:
            message+="+"
            returncode=-3
        
    return (returncode, message, outSMILES)


def process_sdf(isdf):
    pass


def process_smiles(itxt):
    l, n, R = 0, 0, {}
    w = Chem.SDWriter('temp_prefiltered.sdf')
    for line in open(itxt, "r"):
        line=str.split(line.strip(), ";")
        if l==0:
            header=line
        else:
            SMILES, molname, response = line[0], line[1], float(line[2])
            
            if settings.LOWRESP!=None and settings.HIGHRESP!=None: R[molname] = convert_response(y=response, lt=settings.LOWRESP, ht=settings.HIGHRESP)
            else: R[molname]=response
            
            code_outSMILES = ob_mol(string=SMILES, moltype="smi")
            if settings.VERBOSE==2: print("%s\t%s\t%s" % (code_outSMILES[0], molname, code_outSMILES[1]))
            elif settings.VERBOSE==1 and code_outSMILES[0]!=0: print("%s\t%s\t%s" % (code_outSMILES[0], molname, code_outSMILES[1]))
            if code_outSMILES[0]==0:
                # RDKit operations
                m = Chem.MolFromSmiles(code_outSMILES[-1])
                m.SetProp("_Name", molname)
                w.write(m)
                n+=1
        l+=1
    ##settings.RESPONSE_DICT=R
    print("\nNumber of input molecules: %s\nFiltered in: %s\nFiltered out: %s\n" % (l-1, n, l-1-n))
    
    add_response_tag(isdf="temp_prefiltered.sdf", response_dict=R, response_name=header[-1])


def run_prefiltering_operations(args):
    settings.INFILE, settings.HIGHMW, settings.LOWMW, settings.LOWNA, settings.HIGHNA, settings.LOWRESP, settings.HIGHRESP = args.infile, args.highmw, args.lowmw, args.lowna, args.highna, args.lowresp, args.highresp
    if settings.INFILE.endswith(".sdf"): process_sdf(isdf=settings.INFILE)
    else: process_smiles(itxt=settings.INFILE)
    

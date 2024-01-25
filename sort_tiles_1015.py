import json
from glob import glob
import os
from argparse import ArgumentParser
import random
import numpy as np
from shutil import copyfile

def extract_stage(metadata):
    stage = metadata['cases'][0]['diagnoses'][0]['tumor_stage']
    stage = stage.replace(" ", "_")
    stage = stage.rstrip("a")
    stage = stage.rstrip("b")
    return stage

def extract_cancer(metadata):
    return metadata['cases'][0]['project']['project_id']

def extract_sample_type(metadata):
    return metadata['cases'][0]['samples'][0]['sample_type']

def sort_cancer_stage_separately(metadata, **kwargs):
    sample_type = extract_sample_type(metadata)
    cancer = extract_cancer(metadata)
    if "Normal" in sample_type:
        stage = sample_type.replace(" ", "_")
    else:
        stage = extract_stage(metadata)

    return os.path.join(cancer, stage)

def sort_cancer_stage(metadata, **kwargs):
    sample_type = extract_sample_type(metadata)
    cancer = extract_cancer(metadata)
    stage = extract_stage(metadata)
    if "Normal" in sample_type:
        return sample_type.replace(" ", "_")
    return cancer + "_" + stage

def sort_type(metadata, **kwargs):
    cancer = extract_cancer(metadata)
    sample_type = extract_sample_type(metadata)
    if "Normal" in sample_type:
        return sample_type.replace(" ", "_")
    return cancer

def sort_cancer_type(metadata, **kwargs):
    sample_type = extract_sample_type(metadata)
    if "Normal" in sample_type:
        return None
    return extract_cancer(metadata)

def sort_cancer_healthy_pairs(metadata, **kwargs):
    sample_type = extract_sample_type(metadata)
    cancer = extract_cancer(metadata)
    if "Normal" in sample_type:
        return os.path.join(cancer, sample_type.replace(" ", "_"))
    return os.path.join(cancer, cancer)

def sort_cancer_healthy(metadata, **kwargs):
    sample_type = extract_sample_type(metadata)
    if "Normal" in sample_type:
        return sample_type.replace(" ", "_")
    return "cancer"

def sort_random(metadata, **kwargs):
    AllOptions = ['TCGA-LUAD', 'TCGA-LUSC', 'Solid_Tissue_Normal']
    return AllOptions[random.randint(0, 2)]

def sort_mutational_burden(metadata, load_dic, **kwargs):
    submitter_id = metadata["cases"][0]["submitter_id"]
    try:
        return load_dic[submitter_id]
    except KeyError:
        return None

def sort_mutation_metastatic(metadata, load_dic, **kwargs):
    sample_type = extract_sample_type(metadata)
    if "Metastatic" in sample_type:
        submitter_id = metadata["cases"][0]["submitter_id"]
        try:
            return load_dic[submitter_id]
        except KeyError:
            return None
    return None

def sort_setonly(metadata, load_dic, **kwargs):
    return 'All'


def sort_location(metadata, load_dic, **kwargs):
    sample_type = extract_sample_type(metadata)
    return sample_type.replace(" ", "_")

def sort_melanoma_POD(metadata, load_dic, **kwargs):
    Response = metadata['Response to Treatment (Best Response)']
    if 'POD' in Response:
        return 'POD'
    else:
        return 'Response'

def sort_melanoma_Toxicity(metadata, load_dic, **kwargs):
    return metadata['Toxicity Observed']

def sort_text(metadata, load_dic, **kwargs):
    return metadata

def copy_svs_lymph_melanoma(metadata, load_dic, **kwargs):
    sample_type = extract_sample_type(metadata)
    if "Metastatic" in sample_type:
        submitter_id = metadata["cases"][0]["diagnoses"][0]["tissue_or_organ_of_origin"]
        if 'c77' in submitter_id:
            try:
                return True
            except KeyError:
                return False
        else:
            return False
    return False

def copy_svs_skin_primtumor(metadata, load_dic, **kwargs):
    sample_type = extract_sample_type(metadata)
    if "Primary" in sample_type:
        submitter_id = metadata["cases"][0]["diagnoses"][0]["tissue_or_organ_of_origin"]
        if 'c44' in submitter_id:
            try:
                return True
            except KeyError:
                return False
        else:
            return False
    return False

def sort_normal_txt(metadata, load_dic, **kwargs):
    submitter_id = metadata["cases"][0]["submitter_id"]
    if submitter_id in load_dic.keys():
        sample_type = extract_sample_type(metadata)
        if "Normal" in sample_type:
            return sample_type.replace(" ", "_")
        else:
            return load_dic[submitter_id].replace(" ", "_")
    else:
        return None


def sort_melanoma_POD_Rec(metadata, load_dic, **kwargs):
    Response = metadata['Response to Treatment (Best Response)']
    if 'POD' in Response:
        return 'POD'
    elif 'PR' in Response:
        return 'Response'
    elif 'CR' in Response:
        return 'Response'
    else:
        return None

def sort_subfolders(metadata, load_dic, **kwargs):
    return 'All'

sort_options = [
    sort_cancer_stage_separately,
    sort_cancer_stage,
    sort_type,
    sort_cancer_type,
    sort_cancer_healthy_pairs,
    sort_cancer_healthy,
    sort_random,
    sort_mutational_burden,
    sort_mutation_metastatic,
    sort_setonly,
    sort_location,
    sort_melanoma_POD,
    sort_melanoma_Toxicity,
    sort_text,
    copy_svs_lymph_melanoma,
    copy_svs_skin_primtumor,
    sort_normal_txt,
    sort_melanoma_POD_Rec,
    sort_subfolders
]

if __name__ == '__main__':
    descr = """
    ***
    """
    parser = ArgumentParser(description=descr)

    parser.add_argument("--SourceFolder", help="path to tiled images", dest='SourceFolder')
    parser.add_argument("--JsonFile", help="path to metadata json file", dest='JsonFile')
    parser.add_argument("--Magnification", help="magnification to use", type=float, dest='Magnification')
    parser.add_argument("--MagDiffAllowed", help="difference allowed on Magnification", type=float,
                        dest='MagDiffAllowed')
    parser.add_argument("--SortingOption", help="see option at the epilog", type=int, dest='SortingOption')
    parser.add_argument("--PercentValid", help="percentage of images for validation (between 0 and 100)", type=float,
                        dest='PercentValid')
    parser.add_argument("--PercentTest", help="percentage of images for testing (between 0 and 100)", type=float,
                        dest='PercentTest')
    parser.add_argument("--PatientID",
                        help="Patient ID is supposed to be the first PatientID characters (integer expected) of the folder in which the pyramidal jpgs are. Slides from same patient will be in same train/test/valid set. This option is ignored if set to 0 or -1 ",
                        type=int, dest='PatientID')
    parser.add_argument("--TMB", help="path to json file with mutational loads; or to BRAF mutations", dest='TMB')
    parser.add_argument("--nSplit", help="integer n: Split into train/test in n different ways", dest='nSplit')
    parser.add_argument("--Balance", help="balance datasets by: 0- tiles (default); 1-slides; 2-patients (must give PatientID)", type=int, dest='Balance')
    parser.add_argument("--outFilenameStats",
                        help="Check if the tile exists in an out_filename_Stats.txt file and copy it only if it True, or is the expLabel option had the highest probability",
                        dest='outFilenameStats')
    parser.add_argument("--expLabel",
                        help="Index of the expected label within the outFilenameStats file (if only True/False is needed, leave this option empty). comma separated string expected",
                        dest='expLabel')
    parser.add_argument("--threshold",
                        help="threshold above which the probability the class should be to be considered as true (if not specified, it would be considered as true if it has the max probability). comma separated string expected",
                        dest='threshold')

    args = parser.parse_args()

    if args.JsonFile is None:
        print("No JsonFile found")
        args.JsonFile = ''

    if args.PatientID is None:
        print("PatientID ignored")
        args.PatientID = 0

    if args.Balance is None:
    	args.Balance = 0

    if args.nSplit is None:
        args.nSplit = 0
    elif int(args.nSplit) > 0:
        args.PercentValid = 100 / int(args.nSplit)
        args.PercentTest = 0

    if args.outFilenameStats is None:
        outFilenameStats_dict = {}
    else:
        outFilenameStats_dict = {}
        if os.path.isfile(args.outFilenameStats):
            print("outFilenameStats found")
            with open(args.outFilenameStats) as f:
                for line in f:
                    basename = line.split()[0]
                    basename = ".".join("_".join(basename.split("_")[1:]).split(".")[:-1])
                    if args.expLabel is None:
                        isTrue = line.split()[1]
                        outFilenameStats_dict[basename] = isTrue
                    else:
                        ExpectedProb = line.split('[')[-1]
                        ExpectedProb = ExpectedProb.split(']')[0]
                        ExpectedProb = ExpectedProb.split()
                        ExpectedProb = np.array(ExpectedProb)
                        ExpectedProb = np.asfarray(ExpectedProb, float)
                        # print(ExpectedProb)
                        # print("labels exp/true:%s, %d" % (args.expLabel, ExpectedProb.argmax()))
                        if args.threshold is None:
                            #outFilenameStats_dict[basename] = str(int(args.expLabel) == ExpectedProb.argmax())
                            tmp = args.expLabel
                            outFilenameStats_dict[basename] = str(str(ExpectedProb.argmax()) in tmp.split(','))
                        else:
                            tmpT = args.threshold
                            tmpL = args.expLabel
                            tmpR = 'False'
                            tmpT = tmpT.split(',')
                            tmpL = tmpL.split(',')
                            for nL in range(len(tmpL)):
                                # print(ExpectedProb[ int(tmpL[nL]) ], float(tmpT[nL]))
                                if ExpectedProb[ int(tmpL[nL]) ] >= float(tmpT[nL]):
                                    tmpR = 'True'
                            outFilenameStats_dict[basename] = tmpR
                # print(outFilenameStats_dict)
        else:
            print("outFilenameStats NOT found")
            exit()

    SourceFolder = os.path.abspath(args.SourceFolder)
    if args.SortingOption in [15, 16]:
        # raw TCGA svs images
        print("sort option 15 or 16")
        imgFolders = glob(os.path.join(SourceFolder, "*.svs"))
        random.shuffle(imgFolders)  # randomize order of images
    elif args.SortingOption in [19]:
        print("sort option 19")
        imgFolders = glob(os.path.join(SourceFolder, "*", "*_files"))
        random.shuffle(imgFolders)  # randomize order of images
        AllNewDirs = [name for name in os.listdir(SourceFolder) if os.path.isdir(name)]
    else:
        print("sort option other than 15, 16, 19")
        imgFolders = glob(os.path.join(SourceFolder, "*_files"))
        random.shuffle(imgFolders)  # randomize order of images

    JsonFile = args.JsonFile
    nameLength = -1
    if '.json' in JsonFile:
        with open(JsonFile) as fid:
            jdata = json.loads(fid.read())
        try:
            jdata = dict((jd['file_name'].replace('.svs', ''), jd) for jd in jdata)
        except:
            jdata = dict((jd['Patient ID'], jd) for jd in jdata)
    elif args.SortingOption in [10, 19]:
        # no sorting, take everything
        jdata = {}
    else:
        with open(JsonFile, "rU") as f:
            jdata = {}
            nameLength = 10000
            for line in f:
                tmp_PID = line.split()[0]
                nameLength = min(nameLength, len(tmp_PID)) 
                jdata[tmp_PID] = line.split()[1]
            print("minimum nameLength is " + str(nameLength))
    print("jdata:")
    print(jdata)
    Magnification = args.Magnification
    MagDiffAllowed = args.MagDiffAllowed

    SortingOption = args.SortingOption - 1  # transform to 0-based index
    try:
        sort_function = sort_options[SortingOption]
    except IndexError:
        raise ValueError("Unknown sort option")

    if args.SortingOption in [15, 16]:
        # raw TCGA svs images
        for cFolderName in imgFolders:
            print("-----------")
            print(cFolderName)

            imgRootName = os.path.basename(cFolderName)
            #imgRootName = imgRootName.rstrip('.svs')
            if imgRootName.endswith('.svs'):
                imgRootName = imgRootName[:-4]

            try:
                image_meta = jdata[imgRootName]
            except KeyError:
                try:
                    image_meta = jdata[imgRootName[:args.PatientID]]
                except KeyError:
                    print("file_name %s not found in metadata" % imgRootName[:args.PatientID])
                    continue
            IsCopy = sort_function(image_meta, load_dic={})
            if IsCopy:
                copyfile(cFolderName, os.path.join(os.getcwd(), imgRootName + '.svs'))

        quit()

    PercentValid = args.PercentValid / 100.
    if not 0 <= PercentValid <= 1:
        raise ValueError("PercentValid is not between 0 and 100")
    PercentTest = args.PercentTest / 100.
    if not 0 <= PercentTest <= 1:
        raise ValueError("PercentTest is not between 0 and 100")
    TMBFile = args.TMB
    mut_load = {}
    if args.SortingOption == 8:
        if TMBFile:
            with open(TMBFile) as fid:
                mut_load = json.loads(fid.read())
        else:
            raise ValueError("For SortingOption = 8 you must specify the --TMB option")
    elif args.SortingOption == 9:
        if TMBFile:
            with open(TMBFile) as fid:
                mut_load = json.loads(fid.read())
        else:
            raise ValueError("For SortingOption = 9 you must specify the --TMB option")
    elif args.SortingOption == 17:
        if TMBFile:
            with open(TMBFile, "rU") as f:
                mut_load = {}
                for line in f:
                    tmp_PID = line.split()[0]
                    mut_load[tmp_PID] = line.split()[1]
        else:
            raise ValueError("For SortingOption = 17 you must specify the --TMB option")

    print("******************")
    Classes = {}
    NbrTilesCateg = {}
    PercentTilesCateg = {}
    NbrImagesCateg = {}
    PercentSlidesCateg = {}
    NbrPatientsCateg = {}
    PercentPatientsCateg = {}
    Patient_set = {}
    NbSlides = 0
    ttv_split = {}
    nbr_valid = {}
    print("imgFolders: " + str(len(imgFolders)))
    print(imgFolders[0:min(10,len(imgFolders))])
    for cFolderName in imgFolders:
        NbSlides += 1
        # if NbSlides > 10:
        #    raise ValueError("debug")
        #    exit()
        #    raise SystemExit
        #    break
        print("**************** starting %s" % cFolderName)
        imgRootName = os.path.basename(cFolderName)
        imgRootName = imgRootName.replace('_files', '')

        if args.SortingOption == 10:
            SubDir = os.path.basename(os.path.normpath(SourceFolder))
        elif args.SortingOption == 19:
            SubDir = os.path.basename(os.path.split(cFolderName)[0])
            for nAllNewDirs in AllNewDirs:
                if nAllNewDirs in cFolderName:
                    SubDir = nAllNewDirs
        else:
            try:
                image_meta = jdata[imgRootName]
            except KeyError:
                try:
                	#image_meta = jdata[imgRootName[:args.PatientID]]
                    image_meta = jdata[imgRootName[:nameLength]]
                except KeyError:
                    try:
                	    image_meta = jdata[imgRootName[:args.PatientID]]
                    except KeyError:
                        print("file_name %s not found in metadata" % imgRootName[:args.PatientID])
                        continue
            SubDir = sort_function(image_meta, load_dic=mut_load)
        print("SubDir is %s" % SubDir)

        if int(args.nSplit) > 0:
            if SubDir is None:
                print("image not valid for this sorting option")
                continue
            # n-fold cross validation
            for nSet in range(int(args.nSplit)):
                SetDir = "set_" + str(nSet)
                if not os.path.exists(SetDir):
                    os.makedirs(SetDir)
                if SubDir is None:
                    print("image not valid for this sorting option")
                    continue
                if not os.path.exists(os.path.join(SetDir, SubDir)):
                    os.makedirs(os.path.join(SetDir, SubDir))
        else:
            SetDir = ""
            if SubDir is None:
                print("image not valid for this sorting option")
                continue
            if not os.path.exists(SubDir):
                os.makedirs(SubDir)
        # print("SubDir is still %s" % SubDir)
        try:
            Classes[SubDir].append(imgRootName)
        except KeyError:
            Classes[SubDir] = [imgRootName]

        AvailMagsDir = [x for x in os.listdir(cFolderName)
                        if os.path.isdir(os.path.join(cFolderName, x))]
        AvailMags = tuple(float(x) for x in AvailMagsDir)
        if max(AvailMags) < 0:
            print("Magnification was not known for that file.")
            continue
        mismatch, imin = min((abs(x - Magnification), i) for i, x in enumerate(AvailMags))
        if mismatch <= MagDiffAllowed:
            AvailMagsDir = AvailMagsDir[imin]
        else:
            print("No Tiles found at the mag within the allowed range.")
            continue

        print("Symlinking tiles... for subdir %s" % SubDir)
        SourceImageDir = os.path.join(cFolderName, AvailMagsDir, "*")
        AllTiles = glob(SourceImageDir)

        if SubDir in NbrTilesCateg.keys():
            print("%s Already in dictionary" % SubDir)
            # print(SubDir)
        else:
            # print("Not yet in dictionary:")
            # print(SubDir)
            NbrTilesCateg[SubDir] = 0
            NbrTilesCateg[SubDir + "_train"] = 0
            NbrTilesCateg[SubDir + "_test"] = 0
            NbrTilesCateg[SubDir + "_valid"] = 0
            PercentTilesCateg[SubDir + "_train"] = 0
            PercentTilesCateg[SubDir + "_test"] = 0
            PercentTilesCateg[SubDir + "_valid"] = 0
            NbrImagesCateg[SubDir] = 0
            NbrImagesCateg[SubDir + "_train"] = 0
            NbrImagesCateg[SubDir + "_test"] = 0
            NbrImagesCateg[SubDir + "_valid"] = 0
            PercentSlidesCateg[SubDir + "_train"] = 0
            PercentSlidesCateg[SubDir + "_test"] = 0
            PercentSlidesCateg[SubDir + "_valid"] = 0
            NbrPatientsCateg[SubDir + "_NameList"] = {}
            NbrPatientsCateg[SubDir] = 0
            NbrPatientsCateg[SubDir + "_train"] = 0
            NbrPatientsCateg[SubDir + "_test"] = 0
            NbrPatientsCateg[SubDir + "_valid"] = 0            
            PercentPatientsCateg[SubDir + "_train"] = 0
            PercentPatientsCateg[SubDir + "_test"] = 0
            PercentPatientsCateg[SubDir + "_valid"] = 0

            if int(args.nSplit) > 0:
                ttv_split[SubDir] = []
                nbr_valid[SubDir] = []
                for nSet in range(int(args.nSplit)):
                    ttv_split[SubDir].append("train")
                    nbr_valid[SubDir].append(0)
                ttv_split[SubDir][0] = "test"

        NbTiles = 0
        ttv = 'None'
        print("nbr images: " + str(len(AllTiles)))
        if len(AllTiles) == 0:
            continue
        for TilePath in AllTiles:
            # ttv = 'None'          
            TileName = os.path.basename(TilePath)
            # print("TileName is %s" % TileName)
            if len(outFilenameStats_dict) > 0:
                # process only if this tile was classified  as "True" by the classifier
                ThisKey = imgRootName + "_" + TileName.split(".")[0]
                # print(ThisKey)
                if ThisKey in outFilenameStats_dict.keys():
                    if 'False' in outFilenameStats_dict[ThisKey]:
                        continue
                else:
                    continue
            NbTiles += 1
            if args.Balance == 1:
                # print("current percent in test, valid and ID (bal slide):" +  str(PercentSlidesCateg.get(SubDir + "_test"))+ "; " +str(PercentSlidesCateg.get(SubDir + "_valid")))
                # print(PercentTest, PercentValid)
                # print(PercentSlidesCateg.get(SubDir + "_test") < PercentTest)
                # print(PercentSlidesCateg.get(SubDir + "_valid") < PercentValid)
                if (PercentSlidesCateg.get(SubDir + "_test") <= PercentTest) and (PercentTest > 0):
                    ttv = "test"
                elif (PercentSlidesCateg.get(SubDir + "_valid") <= PercentValid) and (PercentValid > 0):
                    ttv = "valid"
                else:
                    ttv = "train"
            elif args.Balance == 2:
                # print("current percent in test, valid and ID (bal patient):" +  str(PercentPatientsCateg.get(SubDir + "_test"))+ "; " +str(PercentPatientsCateg.get(SubDir + "_valid")))
                # print(PercentPatientsCateg.get(SubDir + "_test"))
                # print(PercentPatientsCateg.get(SubDir + "_valid"))
                # print(PercentTest, PercentValid)
                # print(PercentPatientsCateg.get(SubDir + "_test") < PercentTest)
                # print(PercentPatientsCateg.get(SubDir + "_valid") < PercentValid)

                if (PercentPatientsCateg.get(SubDir + "_test") <= PercentTest) and (PercentTest > 0):
                    ttv = "test"
                elif (PercentPatientsCateg.get(SubDir + "_valid") <= PercentValid) and (PercentValid > 0):
                    ttv = "valid"
                else:
                    ttv = "train"
            else :
                # print("current percent in test, valid and ID (bal tile):" +  str(PercentTilesCateg.get(SubDir + "_test"))+ "; " +str(PercentTilesCateg.get(SubDir + "_valid")))
                # print(PercentTilesCateg.get(SubDir + "_test"))
                # print(PercentTilesCateg.get(SubDir + "_valid"))
                # print(PercentTest, PercentValid)
                # print(PercentTilesCateg.get(SubDir + "_test") < PercentTest)
                # print(PercentTilesCateg.get(SubDir + "_valid") < PercentValid)

                if (PercentTilesCateg.get(SubDir + "_test") <= PercentTest) and (PercentTest > 0):
                    ttv = "test"
                elif (PercentTilesCateg.get(SubDir + "_valid") <= PercentValid) and (PercentValid > 0):
                    ttv = "valid"
                else:
                    ttv = "train"

            if int(args.nSplit) > 0:
                for nSet in range(int(args.nSplit)):
                    ttv_split[SubDir][nSet] = "train"

                if args.PatientID > 0:
                    Patient = imgRootName[:args.PatientID]
                    if Patient in Patient_set:
                        SetIndx = Patient_set[Patient]
                        tileNewPatient = False
                    else:
                        SetIndx = nbr_valid[SubDir].index(min(nbr_valid[SubDir]))
                        Patient_set[Patient] = SetIndx
                        tileNewPatient = True
                else:
                    try:
                        Patient = imgRootName
                    except:
                        Patient = imgRootName[:nameLength]
                    if Patient in Patient_set:
                        SetIndx = Patient_set[Patient]
                        tileNewPatient = False
                    else:
                        SetIndx = nbr_valid[SubDir].index(min(nbr_valid[SubDir]))
                        Patient_set[Patient] = SetIndx
                        tileNewPatient = True

                ttv_split[SubDir][SetIndx] = "test"
                if NbTiles == 1:
                    NewPatient = tileNewPatient

                if args.Balance == 1:
                    if NbTiles == 1:
                        nbr_valid[SubDir][SetIndx] = nbr_valid[SubDir][SetIndx] + 1
                elif args.Balance == 2:
                    if NewPatient:
                        if NbTiles == 1:
                            nbr_valid[SubDir][SetIndx] = nbr_valid[SubDir][SetIndx] + 1
                else:
                    nbr_valid[SubDir][SetIndx] = nbr_valid[SubDir][SetIndx] + 1

                for nSet in range(int(args.nSplit)):
                    SetDir = "set_" + str(nSet)
                    NewImageDir = os.path.join(SetDir, SubDir, "_".join(
                        (ttv_split[SubDir][nSet], imgRootName, TileName)))  # all train initially
                    os.symlink(TilePath, NewImageDir)

            else:
                if args.PatientID > 0:
                    Patient = imgRootName[:args.PatientID]
                else:
                    Patient = imgRootName
                if True:
                    if Patient not in NbrPatientsCateg[SubDir + "_NameList"].keys():
                        if Patient in Patient_set:
                            ttv = Patient_set[Patient]
                            NbrPatientsCateg[SubDir + "_NameList"][Patient] = Patient_set[Patient]
                        else:
                            Patient_set[Patient] = ttv
                            NbrPatientsCateg[SubDir + "_NameList"][Patient] = ttv
                        if NbTiles == 1:
                        	NewPatient = True

                    else:
                        ttv = Patient_set[Patient]
                        if NbTiles == 1:
                            NewPatient = False

                NewImageDir = os.path.join(SubDir, "_".join((ttv, imgRootName, TileName)))  # all train initially
                if not os.path.lexists(NewImageDir):
                    os.symlink(TilePath, NewImageDir)

        if ttv == "train":
            if NewPatient: 
                NbrPatientsCateg[SubDir + "_train"] = NbrPatientsCateg[SubDir + "_train"] + 1
            NbrTilesCateg[SubDir + "_train"] = NbrTilesCateg.get(SubDir + "_train") + NbTiles
            NbrImagesCateg[SubDir + "_train"] = NbrImagesCateg[SubDir + "_train"] + 1
        elif ttv == "test":
            if NewPatient: 
                NbrPatientsCateg[SubDir + "_test"] = NbrPatientsCateg[SubDir + "_test"] + 1
            NbrTilesCateg[SubDir + "_test"] = NbrTilesCateg.get(SubDir + "_test") + NbTiles
            NbrImagesCateg[SubDir + "_test"] = NbrImagesCateg[SubDir + "_test"] + 1
        elif ttv == "valid":
            if NewPatient: 
                NbrPatientsCateg[SubDir + "_valid"] = NbrPatientsCateg[SubDir + "_valid"] + 1
            NbrTilesCateg[SubDir + "_valid"] = NbrTilesCateg.get(SubDir + "_valid") + NbTiles
            NbrImagesCateg[SubDir + "_valid"] = NbrImagesCateg[SubDir + "_valid"] + 1
        else:
            continue
        NbrTilesCateg[SubDir] = NbrTilesCateg.get(SubDir) + NbTiles
        NbrImagesCateg[SubDir] = NbrImagesCateg.get(SubDir) + 1
        if NewPatient: 
            NbrPatientsCateg[SubDir] = NbrPatientsCateg.get(SubDir) + 1

        print("New Patient: " + str(NewPatient))
        print("NbrPatientsCateg[SubDir]: " + str(NbrPatientsCateg[SubDir]))
        print("imgRootName: " + str(imgRootName))

        PercentTilesCateg[SubDir + "_train"] = float(NbrTilesCateg.get(SubDir + "_train")) / float(NbrTilesCateg.get(SubDir))
        PercentTilesCateg[SubDir + "_test"] = float(NbrTilesCateg.get(SubDir + "_test")) / float(NbrTilesCateg.get(SubDir))
        PercentTilesCateg[SubDir + "_valid"] = float(NbrTilesCateg.get(SubDir + "_valid")) / float(NbrTilesCateg.get(SubDir))
        PercentSlidesCateg[SubDir + "_train"] = float(NbrImagesCateg.get(SubDir + "_train")) / float(NbrImagesCateg.get(SubDir))
        PercentSlidesCateg[SubDir + "_test"] = float(NbrImagesCateg.get(SubDir + "_test")) / float(NbrImagesCateg.get(SubDir))
        PercentSlidesCateg[SubDir + "_valid"] = float(NbrImagesCateg.get(SubDir + "_valid")) / float(NbrImagesCateg.get(SubDir))
        PercentPatientsCateg[SubDir + "_train"] = float(NbrPatientsCateg.get(SubDir + "_train")) / float(NbrPatientsCateg.get(SubDir))
        PercentPatientsCateg[SubDir + "_test"] = float(NbrPatientsCateg.get(SubDir + "_test")) / float(NbrPatientsCateg.get(SubDir))
        PercentPatientsCateg[SubDir + "_valid"] = float(NbrPatientsCateg.get(SubDir + "_valid")) / float(NbrPatientsCateg.get(SubDir))

        print("Done. %d tiles linked to %s " % (NbTiles, SubDir))
        print("Train / Test / Validation tiles sets for %s = %f %%  / %f %% / %f %%" % (
            SubDir, PercentTilesCateg.get(SubDir + "_train"), PercentTilesCateg.get(SubDir + "_test"),
            PercentTilesCateg.get(SubDir + "_valid")))
        print("Train / Test / Validation slides sets for %s = %f %%  / %f %% / %f %%" % (
            SubDir, PercentSlidesCateg.get(SubDir + "_train"), PercentSlidesCateg.get(SubDir + "_test"),
            PercentSlidesCateg.get(SubDir + "_valid")))
        if args.PatientID > 0:
            print("Train / Test / Validation patients sets for %s = %f %%  / %f %% / %f %%" % (
                SubDir, PercentPatientsCateg.get(SubDir + "_train"), PercentPatientsCateg.get(SubDir + "_test"),
                PercentPatientsCateg.get(SubDir + "_valid")))

    for k, v in sorted(Classes.items()):
        print('list of images in class %s :' % k)
        print(v)

    for k, v in sorted(NbrTilesCateg.items()):
        print(k, v)
    for k, v in sorted(PercentTilesCateg.items()):
        print(k, v)
    for k, v in sorted(NbrImagesCateg.items()):
        print(k, v)
    if args.PatientID > 0:
        for k, v in sorted(NbrPatientsCateg.items()):
            print(k, v)

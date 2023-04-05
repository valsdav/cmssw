### HLT customization functions for 2023 Run-3 studies

customizeHLTFor2023_v3_fromCondDb includes:
- new 2023 HCAL PF rechits thresholds
- new 2023 HCAL rechits thresholds in CaloJets (hltTowerMakerForAll)
- new 2023 PF hadron calibration
- new 2023 Jet Energy Correction for PF jets (AK4PFHLT, AK8PFHLT)
- new 2022 Jet Energy Correction for Calo jets (AK4CaloHLT, AK8CaloHLT)

customizeHLTFor2023_v3 is identical to customizeHLTFor2023_v3_fromCondDb but the calibration and corrections are taken from an external file instead of using conddb.

The OBSOLETE customization functions are as follows.

customizeHLTFor2023_v3_NoCaloJEC:
- new 2023 HCAL PF rechits thresholds
- new 2023 HCAL rechits thresholds in CaloJets (hltTowerMakerForAll)
- new 2023 PF hadron calibration
- new 2023 Jet Energy Correction for PF jets (AK4PFHLT, AK8PFHLT)
- old 2022 Jet Energy Correction for Calo jets (AK4CaloHLT, AK8CaloHLT) --> to be updated

customizeHLTFor2023_v2:
- new 2023 HCAL PF rechits thresholds
- old 2022 HCAL rechits thresholds in CaloJets (hltTowerMakerForAll) --> WRONG! 
- new 2023 PF hadron calibration
- new 2023 Jet Energy Correction for PF jets (AK4PFHLT, AK8PFHLT)
- bugged 2023 Jet Energy Correction for Calo jets (AK4CaloHLT, AK8CaloHLT) obtained with the old 2022 HCAL rechits thresholds in CaloJets --> WRONG!

customizeJECFor2023_noAK8CaloHLT:
- new 2023 HCAL PF rechits thresholds
- old 2022 HCAL rechits thresholds in CaloJets (hltTowerMakerForAll) --> WRONG! 
- new 2023 PF hadron calibration
- new 2023 Jet Energy Correction for PF jets (AK4PFHLT, AK8PFHLT)
- bugged 2023 Jet Energy Correction for Calo jets obtained with the old 2022 HCAL rechits thresholds in CaloJets, only for AK4CaloHLT --> WRONG!


```
cmsrel CMSSW_13_0_0
cd CMSSW_13_0_0/src
cmsenv
git cms-merge-topic silviodonato:customizeHLTFor2023
scram b -j4
hltGetConfiguration (....) > hlt.py 
```

then you can call the customization function(s) by adding at the bottom of your `hlt.py`, as usual. Example:

```
from HLTrigger.Configuration.customizeHLTFor2023 import customizeHLTFor2023_v3
process = customizeHLTFor2023_v3(process)
```

or you can use directly
```
hltGetConfiguration (...) --customise HLTrigger/Configuration/customizeHLTFor2023.customizeHLTFor2023_v3
```

or you can use it in cmsDriver:
```
cmsDriver.py step2 (...) --customise HLTrigger/Configuration/customizeHLTFor2023.customizeHLTFor2023_v3
```

Example 1:
```
hltGetConfiguration /dev/CMSSW_13_0_0/GRun/V24 \
   --globaltag 126X_dataRun3_HLT_v1 \
   --data \
   --unprescale \
   --output minimal \
   --max-events 100 \
   --customise HLTrigger/Configuration/customizeHLTFor2023.customizeHLTFor2023_v3 \
   --eras Run3 \
   --input /store/data/Run2022G/EphemeralHLTPhysics3/RAW/v1/000/362/720/00000/850a6b3c-6eef-424c-9dad-da1e678188f3.root \
   > hltData.py
   
cmsRun hltData.py >& log &
```

Example 2: (taken from step2 of `runTheMatrix.py -l 140.003 -n --extended`)
```
cmsDriver.py step2 --process reHLT -s L1REPACK:Full,HLT:GRun --conditions auto:run3_hlt_relval --data --eventcontent FEVTDEBUGHLT --datatier FEVTDEBUGHLT --era Run3 -n 10 --filein=/store/data/Run2022G/EphemeralHLTPhysics3/RAW/v1/000/362/720/00000/850a6b3c-6eef-424c-9dad-da1e678188f3.root --customise HLTrigger/Configuration/customizeHLTFor2023.customizeHCALFor2023

cmsRun step2_L1REPACK_HLT.py >& log &
```

Example-1 extracts from ConfDB the menu `GRun/V24`, which is the last version of the GRun menu compatible with the last L1T menu of 2022, i.e. [`L1Menu_Collisions2022_v1_4_0`](https://htmlpreview.github.io/?https://github.com/cms-l1-dpg/L1MenuRun3/blob/57fcce7ecf26366084813755f769f47be58bbf5f/development/L1Menu_Collisions2022_v1_4_0/L1Menu_Collisions2022_v1_4_0.html).

Example-2 uses the version of the GRun menu available in the CMSSW release; for `CMSSW_13_0_0`, this corresponds to `GRun/V14`.

### Obsolete customization functions

The OBSOLETE customization functions are:

customizeHLTFor2023_v2 (OBSOLETE):
- new 2023 HCAL PF rechits thresholds
- old 2022 HCAL rechits thresholds in CaloJets (hltTowerMakerForAll) --> WRONG! 
- new 2023 PF hadron calibration
- new 2023 Jet Energy Correction for PF jets (AK4PFHLT, AK8PFHLT)
- new 2023 Jet Energy Correction for Calo jets (AK4CaloHLT, AK8CaloHLT) obtained with the old 2022 HCAL rechits thresholds in CaloJets --> WRONG!

customizeJECFor2023_noAK8CaloHLT (OBSOLETE):
- new 2023 HCAL PF rechits thresholds
- old 2022 HCAL rechits thresholds in CaloJets (hltTowerMakerForAll) --> WRONG! 
- new 2023 PF hadron calibration
- new 2023 Jet Energy Correction for PF jets (AK4PFHLT, AK8PFHLT)
- new 2023 Jet Energy Correction for Calo jets obtained with the old 2022 HCAL rechits thresholds in CaloJets, only for AK4CaloHLT --> WRONG!


### Customization function to run the latest HLT menu on 2022 data collected with L1Menu_Collisions2022_v1_4_0

The latest version of the GRun menu in ConfDB is compatible with the first L1T menu of 2023, i.e. [`L1Menu_Collisions2023_v1_0_0`](https://htmlpreview.github.io/?https://github.com/cms-l1-dpg/L1MenuRun3/blob/57fcce7ecf26366084813755f769f47be58bbf5f/development/L1Menu_Collisions2023_v1_0_0/L1Menu_Collisions2023_v1_0_0.html).

If you want to run on 2022 data using the old [`L1Menu_Collisions2022_v1_4_0`](https://htmlpreview.github.io/?https://github.com/cms-l1-dpg/L1MenuRun3/blob/57fcce7ecf26366084813755f769f47be58bbf5f/development/L1Menu_Collisions2022_v1_4_0/L1Menu_Collisions2022_v1_4_0.html), you can customise the latest HLT GRun menu to make it compatible with the last L1T menu of 2022 using
```
--customise HLTrigger/Configuration/customizeHLTFor2023.customizeHLTFor2022L1TMenu
```
Example:
```
hltGetConfiguration /dev/CMSSW_13_0_0/GRun \
   --globaltag 126X_dataRun3_HLT_v1 \
   --data \
   --unprescale \
   --output minimal \
   --max-events 100 \
   --customise \
HLTrigger/Configuration/customizeHLTFor2023.customizeHLTFor2023_v3,\
HLTrigger/Configuration/customizeHLTFor2023.customizeHLTFor2022L1TMenu \
   --eras Run3 \
   --input /store/data/Run2022G/EphemeralHLTPhysics3/RAW/v1/000/362/720/00000/850a6b3c-6eef-424c-9dad-da1e678188f3.root \
   > hltData.py
```

### Note

You can use separately the functions `customizeHLTFor2023.customizePFHadronCalibrationFor2023` and `customizeHLTFor2023.customizeHCALFor2023`, `customizeHLTFor2023.customizeJECFor2023_noAK8CaloHLT`, and `customizeHLTFor2023.customizeJECFor2023_v2`.

### References

 - Slide 12 from Salavat's talk: [[Indico link]](https://indico.cern.ch/event/1237252/contributions/5204534)

 - PPD coordination meeting: [[Indico link]](https://indico.cern.ch/event/1251668)

 - Changgi's slides at JetMET trigger (PF hadron calibration): [[Indico link]](https://indico.cern.ch/event/1258851/)

 - Changgi's slides at TSG (JEC): [[Indico link]](https://indico.cern.ch/event/1265018/#36-new-hlt-jec)

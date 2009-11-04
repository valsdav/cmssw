#include "DQMOffline/Trigger/interface/EgHLTTrigTools.h"

#include "FWCore/ParameterSet/interface/Registry.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include <boost/algorithm/string.hpp>
using namespace egHLT;

TrigCodes::TrigBitSet trigTools::getFiltersPassed(const std::vector<std::pair<std::string,int> >& filters,const trigger::TriggerEvent* trigEvt,const std::string& hltTag)
{
  TrigCodes::TrigBitSet evtTrigs;
  for(size_t filterNrInVec=0;filterNrInVec<filters.size();filterNrInVec++){
    size_t filterNrInEvt = trigEvt->filterIndex(edm::InputTag(filters[filterNrInVec].first,"",hltTag).encode());
    const TrigCodes::TrigBitSet filterCode = TrigCodes::getCode(filters[filterNrInVec].first.c_str());
    if(filterNrInEvt<trigEvt->sizeFilters()){ //filter found in event, however this only means that something passed the previous filter
      const trigger::Keys& trigKeys = trigEvt->filterKeys(filterNrInEvt);
      if(static_cast<int>(trigKeys.size())>=filters[filterNrInVec].second){
	evtTrigs |=filterCode; //filter was passed
      }
    }//end check if filter is present
  }//end loop over all filters

  return evtTrigs;

}


//this function runs over all parameter sets for every module that has ever run on an event in this job
//it looks for the specified filter module
//and returns the minimum number of objects required to pass the filter, -1 if its not found
//which is either the ncandcut or MinN parameter in the filter config
//assumption: nobody will ever change MinN or ncandcut without changing the filter name
//as this just picks the first module name and if 2 different versions of HLT were run with the filter having
//a different min obj required in the two versions, this may give the wrong answer
int trigTools::getMinNrObjsRequiredByFilter(const std::string& filterName)
{
 
  //will return out of for loop once its found it to save time
  const edm::pset::Registry* psetRegistry = edm::pset::Registry::instance();
  if(psetRegistry==NULL) return -1;
  for(edm::pset::Registry::const_iterator psetIt=psetRegistry->begin();psetIt!=psetRegistry->end();++psetIt){ //loop over every pset for every module ever run
    const std::map<std::string,edm::Entry>& mapOfPara  = psetIt->second.tbl(); //contains the parameter name and value for all the parameters of the pset
    const std::map<std::string,edm::Entry>::const_iterator itToModLabel = mapOfPara.find("@module_label"); 
    if(itToModLabel!=mapOfPara.end()){
      if(itToModLabel->second.getString()==filterName){ //moduleName is the filter name, we have found filter, we will now return something
	std::map<std::string,edm::Entry>::const_iterator itToCandCut = mapOfPara.find("ncandcut");
	if(itToCandCut!=mapOfPara.end() && itToCandCut->second.typeCode()=='I') return itToCandCut->second.getInt32();
	else{ //checks if MinN exists and is int32, if not return -1
	  itToCandCut = mapOfPara.find("MinN");
	  if(itToCandCut!=mapOfPara.end() && itToCandCut->second.typeCode()=='I') return itToCandCut->second.getInt32();
	  else return -1;
	}
      }
      
    }
  }
  return -1;
}
 
//this looks into the HLT config and fills a sorted vector with the last filter of all HLT triggers
//it assumes this filter is either the last (in the case of ES filters) or second to last in the sequence
void trigTools::getActiveFilters(std::vector<std::string>& activeFilters,const std::string& hltTag)
{
  activeFilters.clear();
  
  HLTConfigProvider hltConfig;
  hltConfig.init(hltTag);

  for(size_t pathNr=0;pathNr<hltConfig.size();pathNr++){
    const std::string& pathName = hltConfig.triggerName(pathNr);
    if(pathName.find("HLT_")==0){ //hlt path as they all start with HLT_XXXX
  
      std::string lastFilter;
      const std::vector<std::string>& filters = hltConfig.moduleLabels(pathNr);
      if(!filters.empty()){
	if(filters.back()=="hltBoolEnd" && filters.size()>=2){
	  activeFilters.push_back(filters[filters.size()-2]); //2nd to last element is the last filter, useally the case as last is hltBool except for ES bits
	}else activeFilters.push_back(filters.back());
      }
    }//end hlt path check
  }//end path loop over

  std::sort(activeFilters.begin(),activeFilters.end());
  
}
 
//this function will filter the inactive filternames
//it assumes the list of active filters is sorted   
//at some point this will be replaced with one line of fancy stl code but I want it to work now :)
void trigTools::filterInactiveTriggers(std::vector<std::string>& namesToFilter,const std::vector<std::string>& activeFilters)
{
  //tempory vector to store the filtered results
  std::vector<std::string> filteredNames;
  
  for(size_t inputFilterNr=0;inputFilterNr<namesToFilter.size();inputFilterNr++){
    if(std::binary_search(activeFilters.begin(),activeFilters.end(),namesToFilter[inputFilterNr])){
      filteredNames.push_back(namesToFilter[inputFilterNr]);
    }
  }
  
  namesToFilter.swap(filteredNames);
}

//input filters have format filter1:filter2, this checks both filters are active, rejects ones where both are not active
void trigTools::filterInactiveTightLooseTriggers(std::vector<std::string>& namesToFilter,const std::vector<std::string>& activeFilters)
{
  //tempory vector to store the filtered results
  std::vector<std::string> filteredNames;
  
  for(size_t inputFilterNr=0;inputFilterNr<namesToFilter.size();inputFilterNr++){
    std::vector<std::string> names;
    boost::split(names,namesToFilter[inputFilterNr],boost::is_any_of(":"));
    if(names.size()!=2) continue; //format incorrect, reject it
    if(std::binary_search(activeFilters.begin(),activeFilters.end(),names[0]) &&
       std::binary_search(activeFilters.begin(),activeFilters.end(),names[1])){ //both filters are valid
      filteredNames.push_back(namesToFilter[inputFilterNr]);
    }
  }
  
  namesToFilter.swap(filteredNames);
}


import copy
from equistore import Labels, TensorMap, TensorBlock
import numpy as np

def TensorMapCombinator(tMap1, tMap2):
    
    '''
    This function is used to combine two TensorMaps with the following same properties:
    
        TensorMap.keys with len(TensorMap.keys) == 1
        TensorMap.block(0).samples.names
        TensorMap.block(0).components
        TensorMap.block(0).properties
        TensorMap.block(0).has_gradient('positions')
        TensorMap.block(0).gradient('positions').samples.names
        TensorMap.block(0).gradient('positions').components
        TensorMap.block(0).gradient('positions').properties
    
    Having confirmed that the above properties are the same, this function will
    create a new TensorMap with the following properties:
    
        NewTensorMap.keys = TensorMap1.keys 
                          = TensorMap2.keys
        NewTensorMap.block(0).values = TensorMap1.block(0).values
                                       + TensorMap2.block(0).values
        NewTensorMap.block(0).samples = TensorMap1.block(0).samples 
                                        + TensorMap2.block(0).samples
                                        where the structure indices of TensorMap2.block(0).samples
                                        will increase by max(TensorMap1.block(0).samples['structure'])+1
        NewTensorMap.block(0).components = TensorMap1.block(0).components
                                         = TensorMap2.block(0).components
        NewTensorMap.block(0).properties = TensorMap1.block(0).properties
                                         = TensorMap2.block(0).properties
        NewTensorMap.block(0).gradient(‘positions’).data = TensorMap1.block(0).gradient(‘positions’).data
                                                         + TensorMap2.block(0).gradient(‘positions’).data
        NewTensorMap.block(0).gradient(‘positions’).samples = xxx.gradient(‘positions’).samples 
                                                            + xxx.gradient(‘positions’).samples
        components are properties of TensorMap.block(0).gradient(‘positions’) remain the same
    
    Have fun!
    '''
    
    ### Examination
    if len(tMap1.keys) != 1:
        raise Exception('len(TensorMap1.keys) should be 1!')
    if len(tMap2.keys) != 1:
        raise Exception('len(TensorMap2.keys) should be 1!')
        
    if tMap1.keys != tMap2.keys:
        raise Exception('The keys of the two TensorMaps should be the same!')
    
    if tMap1.block(0).samples.names != tMap2.block(0).samples.names:
        raise Exception('The names of the two TensorMaps.block(0).samples should be the same!')
    
    if tMap1.block(0).components != tMap2.block(0).components:
        raise Exception('The components of the two TensorMaps.block(0) should be the same!')
    
    if tMap1.block(0).properties.shape != tMap2.block(0).properties.shape:
        raise Exception('The shapes of the two TensorMaps.block(0).properties should be the same!')
    
    for i in np.arange(len(tMap1.block(0).properties)):
        for j in np.arange(len(tMap1.block(0).properties[0])):
            if tMap1.block(0).properties[i][j] != tMap2.block(0).properties[i][j]:
                raise Exception('The TensorMaps.block(0).properties should be exactly the same!')
            else:
                continue
    
    ### New values of the combined TensorMap
    tMapValues = np.concatenate((tMap1.block(0).values,tMap2.block(0).values),axis=0)
    
    ### New samples of the combined TensorMap
    tMap2Temp = copy.deepcopy(tMap2.block(0).samples)
    tMap2Temp['structure'] += max(tMap1.block(0).samples['structure'])+1
    newSamples = np.concatenate((tMap1.block(0).samples,tMap2Temp), axis=0)
    sampleList = []
    for i in newSamples:
        sampleEntry = []
        for j in i:
            sampleEntry.append(j)
        sampleList.append(sampleEntry)
    sampleList = np.array(sampleList)
    tMapSamples = Labels(tMap1.block(0).samples.names, sampleList)
    
    ### New components of the combined TensorMap
    tMapComponents = tMap1.block(0).components
    
    ### New properties of the combined TensorMap
    tMapProperties = tMap1.block(0).properties
    
    
    tMapTensorBlock = TensorBlock(tMapValues, tMapSamples, tMapComponents, tMapProperties)
    
    
    ### Gradient Examination
    if tMap1.block(0).has_gradient('positions') != tMap2.block(0).has_gradient('positions'):
        raise Exception('The TensorMap.block(0).has_gradient("positions") should be the same!')
    
    if tMap1.block(0).has_gradient('positions') == True:
    
        if tMap1.block(0).gradient('positions').samples.names != tMap1.block(0).gradient('positions').samples.names:
            raise Exception('The gradient.samples.names should be the same!')

        ### Examination of gradient components is skipped
        
        for i in np.arange(len(tMap1.block(0).gradient('positions').properties)):
            for j in np.arange(len(tMap1.block(0).gradient('positions').properties[0])):
                if tMap1.block(0).gradient('positions').properties[i][j] != tMap2.block(0).gradient('positions').properties[i][j]:
                    raise Exception('The gradient.properties should be exactly the same!')
                else:
                    continue
        
        ### New gradient data
        gradData = np.concatenate((tMap1.block(0).gradient('positions').data,tMap2.block(0).gradient('positions').data),axis=0)
        
        ### New gradient samples
        grad2Temp = copy.deepcopy(tMap2.block(0).gradient('positions').samples)
        grad2Temp['sample'] += max(tMap1.block(0).gradient('positions').samples['sample'])+1
        grad2Temp['structure'] += max(tMap1.block(0).gradient('positions').samples['structure'])+1
        gradSamples = np.concatenate((tMap1.block(0).gradient('positions').samples,grad2Temp), axis=0)
        gradSampleList = []
        for i in gradSamples:
            gradSampleEntry = []
            for j in i:
                gradSampleEntry.append(j)
            gradSampleList.append(gradSampleEntry)
        gradSampleList = np.array(gradSampleList)
        gradSamples = Labels(tMap1.block(0).gradient('positions').samples.names, gradSampleList)
        
        ### New gradient components
        gradComponents = tMap1.block(0).gradient('positions').components
        
        
        tMapTensorBlock.add_gradient('positions', gradData, gradSamples, gradComponents)
    
    tMap = TensorMap(tMap1.keys, [tMapTensorBlock])
    
    return tMap

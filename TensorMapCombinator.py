def TensorMapCombinator(tMap1, tMap2):
    
    import copy
    from equistore import Labels, TensorMap, TensorBlock
    
    '''
    This function is used to combine two TensorMaps with the following same properties:
    
        TensorMap.keys with len(TensorMap.keys) == 1
        TensorMap.block(0).samples.names
        TensorMap.block(0).components
        TensorMap.block(0).properties
    
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
    
    tMap = TensorMap(tMap1.keys, [tMapTensorBlock])
    
    return tMap

import numpy as np
from equistore import Labels, TensorBlock, TensorMap

def rascalLodeCombinator(lr, srsr, alpha):
    
    '''
    alpha*lr + srsr
    '''
    assert np.all(lr.keys == srsr.keys)

    merged_blocks = []

    for key, srsr_block in srsr:
        lr_block = lr.block(key)
        srsr_block = srsr.block(key)

        assert np.all(lr_block.samples == srsr_block.samples)
        assert lr_block.components == []
        assert srsr_block.components == []

        n_properties = len(lr_block.properties) + len(srsr_block.properties)

        merged_properties = Labels(
            names=["dummy"],
            values=np.arange(n_properties, dtype=np.int32).reshape(-1, 1)
        )

        new_block = TensorBlock(
            values=np.hstack([alpha*lr_block.values, srsr_block.values]),
            samples=lr_block.samples,
            components=[],
            properties=merged_properties,
        )
        
        
        if lr_block.has_gradient('positions') == True:
            
            lr_gradients = lr_block.gradient("positions")
            srsr_gradients = srsr_block.gradient("positions")

            #assert np.all(lr_gradients.components == srsr_gradients.components)

            # just copy lr gradient data over
            new_gradient_data = np.zeros((len(lr_gradients.samples), 3, n_properties))
            new_gradient_data[:, :, :len(lr_block.properties)] = alpha*lr_gradients.data

            # check where to put the srsr data
            for srsr_grad_i, srsr_grad_sample in enumerate(srsr_gradients.samples):
                merged_i = lr_gradients.samples.position(srsr_grad_sample)
                new_gradient_data[merged_i, :, len(lr_block.properties):] = srsr_gradients.data[srsr_grad_i, :, :]

            new_block.add_gradient(
                "positions",
                new_gradient_data,
                lr_gradients.samples,
                lr_gradients.components,
            )

        merged_blocks.append(new_block)

    merged_tensor = TensorMap(lr.keys, merged_blocks)
    
    return merged_tensor
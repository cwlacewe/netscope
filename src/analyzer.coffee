module.exports =
class Analyzer
    constructor: ->
        
    analyze: (net) ->
        ## Add Input/Output Dimensions + Channels to each Node / Layer
        # shape.dim: (    N   x   K   x   W   x   H   )
        #              batch   channel  width   height
        #               chIn    chOut   wIn     wOut
    
        for n in net.sortTopologically()
            # init to zero
            d = n.analysis
            d.wIn  = d.hIn = d.wOut = d.hOut = 0
            d.chIn = d.chOut = 0
            d.comp = {macc: 0, comp: 0, add: 0, div: 0, exp: 0}
            d.mem  = {activation: 0, param: 0}
            d.variants = [];
        
            layertype = n.type.toLowerCase()
            parent = n.parents[0]?.analysis
                
            switch layertype
                when "data"
                    #dimensions
                    if n.attribs.input_param?.shape?
                        shape = n.attribs.input_param.shape
                        d.chIn  = shape.dim[1]
                        d.hIn   = shape.dim[2]
                        d.wIn   = shape.dim[3]
                    else if n.attribs.transform_param?.crop_size?
                        d.wIn  = d.hIn = n.attribs.transform_param.crop_size
                        d.chIn = 3
                    else
                        onerror('Unknown Input Dimensions')
                        debugger;
                    d.wOut  = d.wIn
                    d.hOut  = d.hIn
                    d.chOut = d.chIn
                    #computation
                    #-- none
                    #memory
                    #-- none 
                    d.mem.activation = d.wOut*d.hOut*d.chOut
                    
                when "convolution"
                    #dimensions
                    params   = n.attribs.convolution_param
                    kernel_w = params.kernel_w ? params.kernel_size
                    kernel_h = params.kernel_h ? params.kernel_size
                    stride_w = params.stride_w ? (params.stride ? 1)
                    stride_h = params.stride_h ? (params.stride ? 1)
                    pad_w    = params.pad_w ? (params.pad ? 0)
                    pad_h    = params.pad_h ? (params.pad ? 0)
                    numout   = params.num_output
                    d.wIn    = parent.wOut
                    d.hIn    = parent.hOut
                    # according to http://caffe.berkeleyvision.org/tutorial/layers.html and https://github.com/BVLC/caffe/issues/3656 
                    d.wOut = Math.floor((d.wIn + 2*pad_w - kernel_w) / stride_w) + 1
                    d.hOut = Math.floor((d.hIn + 2*pad_h - kernel_h) / stride_h) + 1
                    d.chIn = parent.chOut
                    d.chOut = numout
                    #computation
                    d.comp.macc = (kernel_w*kernel_h)*(d.wOut*d.hOut)*d.chIn*d.chOut
                    #memory
                    d.mem.param = (kernel_w*kernel_h)*d.chIn*d.chOut
                    d.mem.activation = d.wOut*d.hOut*d.chOut
                    
                    # CACHE AND BANDWIDTH for Implementation Variants
                    if (do_variants_analysis)
                        d.variants.push({
                          name  : "complete outputs, input cache"
                          cache : d.chIn*kernel_h*d.wIn +          # line buffers
                                  d.chIn*kernel_h*kernel_w         # param cache
                          readBW  : d.chOut*d.chIn*(d.wIn*d.hIn)
                          writeBW : d.chOut*(d.wOut*d.hOut)          # ideal
                          confBW  : d.chOut*d.chIn*kernel_w*kernel_h # ideal
                        })
                        d.variants.push({
                          name  : "complete inputs, input cache"
                          cache : kernel_h*d.wIn +         # line buffers
                                  d.chIn*kernel_h*kernel_w # param cache
                          readBW  : d.chIn*((d.chOut+1)*(d.wIn*d.hIn))
                          writeBW : d.chIn*((d.chOut)*(d.wOut*d.hOut))
                          confBW  : d.chOut*d.chIn*kernel_w*kernel_h # ideal
                        })
                        d.variants.push({
                          name  : "complete inputs, input + output cache"
                          cache : kernel_h*d.wIn +           # line buffers
                                  d.chIn*kernel_h*kernel_w + # param cache
                                  d.wIn*d.hIn*d.chOut        # output cache
                          readBW  : d.chIn*(d.wIn*d.hIn)  # ideal
                          writeBW : d.chOut*(d.wOut*d.hOut) # ideal
                          confBW  : d.chOut*d.chIn*kernel_w*kernel_h # ideal
                        })
                        d.variants.push({
                          name  : "streaming, input cache"
                          cache : d.chIn*kernel_h*d.wIn
                          readBW  : d.chIn*(d.wIn*d.hIn)
                          writeBW : d.chOut*(d.wOut*d.hOut)
                          confBW  : d.hIn*(d.chIn*d.chOut*(kernel_w*kernel_h))
                        })
                        d.variants.push({
                          name  : "streaming, input + config cache"
                          cache : d.chIn*kernel_h*d.wIn +
                                  d.chIn*d.chOut*(kernel_h*kernel_w)
                          readBW  : d.chIn*(d.wIn*d.hIn)
                          writeBW : d.chOut*(d.wOut*d.hOut)
                          confBW  : d.chOut*d.chIn*kernel_w*kernel_h # ideal
                        })
                        d.variants.push({
                          name  : "streaming, temp."
                          img_cache : if kernel_h > 1 then d.chIn*kernel_h*d.wIn else d.chIn
                          img_dim   : d.chIn+"ch ∙ "+d.wIn+" × "+kernel_h+" × 32b"
                          flt_cache : d.chIn*d.chOut*(kernel_h*kernel_w)
                          squeeze_cache : if n.name.indexOf("squeeze") > -1 then d.chOut*d.wOut*d.hOut else ""
                        })
                    
                
                when "innerproduct", "inner_product"
                    #dimensions
                    numout = n.attribs.inner_product_param.num_output
                    d.wIn  = parent.wOut
                    d.hIn  = parent.hOut
                    d.chIn = parent.chOut
                    d.wOut = 1
                    d.hOut = 1
                    d.chOut = numout
                    #computation
                    d.comp.macc = (d.wIn*d.hIn)*d.chIn*d.chOut
                    #memory
                    d.mem.param = d.wIn*d.hIn*d.chIn*d.chOut
                    d.mem.activation = d.wOut*d.hOut*d.chOut
                
                when "pooling"
                    #dimensions
                    params = n.attribs.pooling_param
                    kernel = params.kernel_size
                    stride = params.stride ? 1
                    pad    = params.pad ? 0
                    isglobal = params.global_pooling ? 0
                    pooltype = (params.pool ? 'MAX').toUpperCase()
                    d.wIn  = parent.wOut
                    d.hIn  = parent.hOut
                    d.chIn = parent.chOut
                    d.chOut = d.chIn
                    # according to http://caffe.berkeleyvision.org/tutorial/layers.html and https://github.com/BVLC/caffe/issues/3656
                    d.wOut = Math.ceil((d.wIn + 2*pad - kernel) / stride) + 1
                    d.hOut = Math.ceil((d.hIn + 2*pad - kernel) / stride) + 1
                    if isglobal
                        d.wOut = d.hOut = 1
                    #computation
                    num_ops = if isglobal then ((d.wIn*d.hIn)*d.chIn) else ((d.wOut*d.hOut)*kernel*kernel*d.chOut)
                    if pooltype == 'MAX'
                        d.comp.comp = num_ops
                    else if pooltype == 'AVE'
                        d.comp.add = num_ops
                        #d.comp.div = (d.wOut*d.hOut*d.chOut) #divide by const.
                    else    
                        onerror "Unknown pooling type #{pooltype}"
                    #memory
                    d.mem.activation = d.wOut*d.hOut*d.chOut
                
                when "batchnorm"
                    #dimensions
                    d.wIn  = parent.wOut
                    d.hIn  = parent.hOut
                    d.wOut = d.wIn
                    d.hOut = d.hIn
                    d.chOut = d.chIn = parent.chOut
                    #computation
                    # BN: subtract mean, divide by variance for each channel
                    # averages during training: over spatial dims + batch
                    d.comp.add = d.wIn*d.hIn*d.chIn
                    d.comp.div = d.wIn*d.hIn*d.chIn
                    #memory
                    d.mem.param = d.chIn*2
                    d.mem.activation = d.wOut*d.hOut*d.chOut
            
                when "lrn"
                    #dimensions
                    #default mode: ACROSS_CHANNELS
                    mode   = n.attribs.lrn_param.norm_region ? 'ACROSS_CHANNELS'
                    size   = n.attribs.lrn_param.local_size
                    d.wIn  = parent.wOut
                    d.hIn  = parent.hOut
                    d.wOut = d.wIn
                    d.hOut = d.hIn
                    d.chOut = d.chIn = parent.chOut
                    #computation
                    #  Each input value is divided by (1+(α/n)∑xi^2)^β
                    num_inputs = d.wIn*d.hIn*d.chIn
                    d.comp.macc = num_inputs*size   # (∑xi^2)
                    d.comp.add = num_inputs         # (1+...)
                    d.comp.exp = num_inputs         # (...)^β
                    d.comp.div = num_inputs*2       # (α/n)*... + divide by sum
                    #memory
                    d.mem.param = 2  # alpha, beta
                    d.mem.activation = d.wOut*d.hOut*d.chOut
                
                when "concat"
                    #dimensions
                    d.wIn = parent.wOut
                    d.hIn = parent.hOut
                    d.wOut = d.wIn
                    d.hOut = d.hIn
                    # sum up channels from inputs
                    d.chIn += p.analysis.chOut for p in n.parents
                    d.chOut = d.chIn
                    # check input dimensions
                    failed = failed || (p.analysis.wOut != d.wIn || p.analysis.hOut != d.hIn) for p in n.parents
                    window.onerror('CONCAT: input dimensions dont agree!') if failed
                    #computation
                    # --none
                    #memory
                    d.mem.activation = d.wOut*d.hOut*d.chOut
                    
                #relu/dropout use some memory, do some comparisons
                when "relu", "dropout"
                    #dimensions
                    d.wIn = parent.wOut
                    d.hIn = parent.hOut
                    d.wOut = d.wIn
                    d.hOut = d.hIn
                    d.chOut = d.chIn = parent.chOut
                    #computation
                    d.comp.comp = d.wIn*d.hIn*d.chIn
                    #memory
                    d.mem.activation = d.wOut*d.hOut*d.chOut
                    
                when "softmax", "softmaxwithloss", "softmax_loss"
                    #dimensions
                    d.wIn = parent.wOut
                    d.hIn = parent.hOut
                    d.wOut = d.wIn
                    d.hOut = d.hIn
                    d.chOut = d.chIn = parent.chOut
                    #computation
                    d.comp.exp = d.wIn*d.hIn*d.chIn
                    d.comp.add = d.wIn*d.hIn*d.chIn
                    d.comp.div = d.wIn*d.hIn*d.chIn
                    #memory
                    d.mem.activation = d.wOut*d.hOut*d.chOut
                    
                when "flatten"
                    #dimensions
                    d.wIn = parent.wOut
                    d.hIn = parent.hOut      
                    d.chIn = parent.chOut
                    d.wOut = d.hOut = 1
                    d.chOut = d.chIn * d.wIn * d.hIn
                    #computation
                    # --none
                    #memory
                    d.mem.activation = d.wOut*d.hOut*d.chOut
                    
                when "eltwise"
                    #dimensions
                    d.wIn = parent.wOut
                    d.hIn = parent.hOut      
                    d.chIn = parent.chOut
                    d.wOut = d.wIn
                    d.hOut = d.hIn
                    d.chOut = d.chIn
                    # check input dimensions
                    parent2 = n.parents[1].analysis
                    failed = parent.wOut != parent2.wOut or parent.hOut != parent2.hOut
                    onerror 'ELTWISE: input dimensions dont agree!' if failed
                    #computation
                    op = n.eltwise_param?.operation?.toUpperCase() ? 'SUM'
                    if op == 'SUM'
                        d.comp.add = d.wIn*d.hIn*d.chIn
                    else if op == 'MAX'
                        d.comp.comp = d.wIn*d.hIn*d.chIn
                    else if op == 'PROD'
                        d.comp.macc = d.wIn*d.hIn*d.chIn
                    else
                        onerror 'ELTWISE: unknown operation '+op
                    #memory
                    d.mem.activation = d.wOut*d.hOut*d.chOut
                    
                when "deconvolution"
                    #dimensions
                    params   = n.attribs.convolution_param
                    kernel_w = params.kernel_w ? params.kernel_size
                    kernel_h = params.kernel_h ? params.kernel_size
                    stride_w = params.stride_w ? (params.stride ? 1)
                    stride_h = params.stride_h ? (params.stride ? 1)
                    pad_w    = params.pad_w ? (params.pad ? 0)
                    pad_h    = params.pad_h ? (params.pad ? 0)
                    numout   = params.num_output
                    d.wIn = parent?.wOut
                    d.hIn = parent?.hOut      
                    d.chIn = parent?.chOut
                    d.wOut = (stride_w*(d.wIn-1)+kernel_w-2*pad_w)
                    d.hOut = (stride_h*(d.hIn-1)+kernel_h-2*pad_h)
                    d.chOut = numout
                    #computation
                    d.comp.macc = d.chIn*d.chOut*d.wOut*d.hOut*(kernel_w/stride_w)*(kernel_h/stride_h)
                    #memory
                    d.mem.param = kernel_w*kernel_h*d.chIn*d.chOut
                    d.mem.activation = d.wOut*d.hOut*d.chOut
                    
                when "crop"
                    #dimensions
                    ## crop to dims of 2nd parent
                    parent2 = n.parents[1].analysis
                    d.wIn = parent?.wOut
                    d.hIn = parent?.hOut
                    d.chIn = parent?.chOut
                    d.wOut = parent2.wOut
                    d.hOut = parent2.hOut
                    d.chOut = d.chIn
                    #computation
                    # --none
                    #memory
                    d.mem.activation = d.wOut*d.hOut*d.chOut
                          
                #scale layer use activation memory and does multiplies
                when "scale"
                    #dimensions
                    ## assume pass-through
                    d.wIn = parent?.wOut
                    d.hIn = parent?.hOut
                    d.chIn = parent?.chOut
                    d.wOut = d.wIn
                    d.hOut = d.hIn
                    d.chOut = d.chIn
                    #computation: scale = multiplication
                    d.comp.macc = d.wOut*d.hOut*d.chOut
                    #memory
                    d.mem.activation = d.wOut*d.hOut*d.chOut
                                   
                #implicit layers use activation memory, but no computation 
                when "implicit"
                    #dimensions
                    ## assume pass-through
                    d.wIn = +parent?.wOut?
                    d.hIn = +parent?.hOut?
                    d.chIn = +parent?.chOut?
                    d.wOut = d.wIn
                    d.hOut = d.hIn
                    d.chOut = d.chIn
                    #computation
                    # --none
                    #memory
                    d.mem.activation = d.wOut*d.hOut*d.chOut
                    
                # accuracy layers just pass through
                when "accuracy"
                    #dimensions
                    ## assume pass-through
                    d.wIn = parent?.wOut
                    d.hIn = parent?.hOut
                    d.chIn = parent?.chOut
                    d.wOut = d.wIn
                    d.hOut = d.hIn
                    d.chOut = d.chIn
                    #computation
                    # --none
                    #memory
                    # --none

                # reshape layers just permute dimensions, assume on-the-fly operation
                when "reshape"
                    #get reshape parameters
                    newshape = n.attribs.reshape_param.shape.dim.slice(0) # copy array
                    #debugger
                    console.log(newshape);
                    # 0 as dimension = inherit from input
                    if (not newshape[0]) or (newshape[0] == 0) then newshape[0] = d.batchIn
                    if (not newshape[1]) or (newshape[1] == 0) then newshape[1] = d.chIn
                    if (not newshape[2]) or (newshape[2] == 0) then newshape[2] = d.hIn
                    if (not newshape[3]) or (newshape[3] == 0) then newshape[3] = d.wIn
                    # -1 as dimension = infer from other dimensions, allowed for at most 1 dimension
                    prod_in_dims = d.batchIn * d.wIn * d.hIn * d.chIn
                    prod_out_dims = newshape[0] * newshape[1] * newshape[2] * newshape[3] * (-1)# -1 compensates "-1" in newshape
                    infered_dim = prod_in_dims / prod_out_dims
                    if newshape[0] == -1 then newshape[0] = infered_dim
                    if newshape[1] == -1 then newshape[1] = infered_dim
                    if newshape[2] == -1 then newshape[2] = infered_dim
                    if newshape[3] == -1 then newshape[3] = infered_dim
                    # assign output dimensions
                    d.batchOut = newshape[0]
                    d.chOut    = newshape[1]
                    d.hOut     = newshape[2]
                    d.wOut     = newshape[3]
                    # no computation, no memory required.
                
                else # unknown layer;  print error message;
                    onerror('Unknown Layer: '+layertype)
                    console.log(n)
                    debugger;

            # add dimensions to node attributes so they show in graph tooltips
            trivial_layers = ["softmax", "softmaxwithloss", "softmax_loss", "dropout", "concat", "accuracy"]
            if $.inArray(layertype, trivial_layers) == -1
                analysis = {
                    in: d.chIn+'ch ⋅ '+d.wIn+'×'+d.hIn,
                    out: d.chOut+'ch ⋅ '+d.wOut+'×'+d.hOut }
                ops = (val+'⋅'+key for key,val of d.comp when val isnt 0).join(', ')
                #debugger
                analysis.ops = ops if ops != ""
                mem = (val+'⋅'+key for key,val of d.mem when val isnt 0).join(', ')
                analysis.mem = mem if mem != ""
                _.extend(n.attribs, {analysis: analysis});
                
        return net
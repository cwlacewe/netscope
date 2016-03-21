Parser = require './parser'
Network = require '../network.coffee'

generateLayers = (descriptors, phase) ->
    phase ?= 'train'
    layers = []
    for entry in descriptors
        # Support the deprecated Caffe 'layers' key as well.
        layerDesc = entry.layer or entry.layers
        if layerDesc?
            layer = {}
            headerKeys = ['name', 'type', 'top', 'bottom']
            _.extend layer, _.pick(layerDesc, headerKeys)
            layer.attribs = _.omit layerDesc, headerKeys
            layers.push layer
        else
            console.log 'Unidentified entry ignored: ', entry
    layers = _.filter layers, (layer) ->
        layerPhase = layer.attribs.include?.phase
        not (layerPhase? and layerPhase!=phase)
    return layers

analyzeNetwork = (net) ->
    ## Add Input/Output Dimensions + Channels to each Node / Layer
    # shape.dim: (    N   x   K   x   W   x   H   )
    #              batch   channel  width   height
    #               chIn    chOut   wIn     wOut
    
    for n in net.nodes
        # init to zero
        d = n.analysis
        d.wIn = d.hIn = d.wOut = d.hOut = 0
        d.chIn = d.chOut = 0
        
        layertype = n.type.toUpperCase()
        parent = n.parents[0]?.analysis
                
        switch layertype
            when "DATA"
                if (n.attribs.input_param?)
                    d.chIn = n.attribs.input_param.shape.dim[1]
                    d.wIn = n.attribs.input_param.shape.dim[2]
                    d.hIn = n.attribs.input_param.shape.dim[3]
                else if (n.attribs.transform_param?.crop_size?)
                    d.wIn = d.hIn = n.attribs.transform_param.crop_size
                    d.chIn = 3
                else
                    onerror('Unknown Input Dimensions')
                    debugger;
                d.wOut = d.wIn
                d.hOut = d.hIn
                d.chOut = d.chIn
                
            when "CONVOLUTION"
                kernel_w = n.attribs.convolution_param.kernel_w ? n.attribs.convolution_param.kernel_size
                kernel_h = n.attribs.convolution_param.kernel_h ? n.attribs.convolution_param.kernel_size
                stride_w = n.attribs.convolution_param.stride_w ? (n.attribs.convolution_param.stride ? 1)
                stride_h = n.attribs.convolution_param.stride_h ? (n.attribs.convolution_param.stride ? 1)
                pad_w  = n.attribs.convolution_param.pad_w ? (n.attribs.convolution_param.pad ? 0)
                pad_h  = n.attribs.convolution_param.pad_h ? (n.attribs.convolution_param.pad ? 0)
                numout = n.attribs.convolution_param.num_output
                d.wIn  = parent.wOut
                d.hIn  = parent.hOut
                # according to http://caffe.berkeleyvision.org/tutorial/layers.html and https://github.com/BVLC/caffe/issues/3656 
                d.wOut = Math.floor((d.wIn + 2*pad_w - kernel_w) / stride_w) + 1
                d.hOut = Math.floor((d.hIn + 2*pad_h - kernel_h) / stride_h) + 1
                d.chIn = parent.chOut
                d.chOut = numout
                
            when "INNERPRODUCT"
                numout = n.attribs.inner_product_param.num_output
                d.wIn  = parent.wOut
                d.hIn  = parent.hOut
                d.wOut = d.wIn
                d.hOut = d.hIn
                d.chIn = parent.chOut
                d.chOut = numout
                
            when "POOLING"
                kernel = n.attribs.pooling_param.kernel_size
                stride = n.attribs.pooling_param.stride ? 1
                pad    = n.attribs.pooling_param.pad ? 0
                isglobal = n.attribs.pooling_param.global_pooling ? 0
                d.wIn = parent.wOut
                d.hIn = parent.hOut
                # according to http://caffe.berkeleyvision.org/tutorial/layers.html and https://github.com/BVLC/caffe/issues/3656
                if isglobal
                    d.wOut = d.hOut = 1
                else
                    d.wOut = Math.ceil((d.wIn + 2*pad - kernel) / stride) + 1
                    d.hOut = Math.ceil((d.hIn + 2*pad - kernel) / stride) + 1                
                d.chOut = d.chIn = parent.chOut
                
            when "BATCHNORM"
                d.wIn  = parent.wOut
                d.hIn  = parent.hOut
                d.wOut = d.wIn
                d.hOut = d.hIn
                d.chOut = d.chIn = parent.chOut
            
            when "LRN"
                #default mode: ACROSS_CHANNELS
                mode   = n.attribs.lrn_param.norm_region ? 'ACROSS_CHANNELS'
                size   = n.attribs.lrn_param.local_size
                d.wIn  = parent.wOut
                d.hIn  = parent.hOut
                d.wOut = d.wIn
                d.hOut = d.hIn
                d.chOut = d.chIn = parent.chOut
                
            when "CONCAT"
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
     
            when "RELU", "DROPOUT", "SOFTMAX", "SOFTMAXWITHLOSS"
                d.wIn = parent.wOut
                d.hIn = parent.hOut
                d.wOut = d.wIn
                d.hOut = d.hIn
                d.chOut = d.chIn = parent.chOut
                
            when "FLATTEN"
                d.wIn = parent.wOut
                d.hIn = parent.hOut      
                d.chIn = parent.chOut
                d.wOut = d.hOut = 1
                d.chOut = d.chIn * d.wIn * d.hIn
                
            when "IMPLICIT"
                debugger;
                d.wIn = parent?.wOut ? 0
                d.hIn = parent?.hOut ? 0
                d.wOut = 0
                d.hOut = 0
                d.chIn = parent?.chOut
                d.chOut = 0
                
            else # unknown layer;  print error message;
                onerror('Unknown Layer: '+layertype)
                console.log(n)
                debugger;

        # add dimensions to node attributes
        # so they show in graph tooltips
        if (layertype!="RELU" && layertype!="SOFTMAX" && layertype!="SOFTMAXWITHLOSS")
            _.extend(n.attribs, {
            analysis: {
                in: d.chIn+'ch ⋅ '+d.wIn+'×'+d.hIn,
                out: d.chOut+'ch ⋅ '+d.wOut+'×'+d.hOut
                }} )
                
    return net

generateNetwork = (layers, header) ->
    nodeTable = {}
    implicitLayers = []
    net = new Network header.name
    getSingleNode = (name) =>
        node = nodeTable[name]
        # Caffe allows top to be a layer which isn't explicitly
        # defined. Create an implicit layer if this is detected.
        if not node?
            debugger;
            node = net.createNode name, 'implicit'
            nodeTable[name] = node
        return node
    getNodes = (names, exclude) =>
        names = [].concat names
        if exclude?
            _.pullAll names, exclude
        _.map names, getSingleNode
    # Build the node LUT.
    for layer in layers
        nodeTable[layer.name] = net.createNode layer.name, layer.type, layer.attribs, {}
    # Connect layers.
    inplaceTable = {}
    for layer in layers
        node = nodeTable[layer.name]
        if layer.top?
            if layer.top==layer.bottom
                # This is an inplace node. We will treat this specially.
                # Note that this would have otherwise introduced a cycle,
                # violating the requirements of a DAG.
                if not inplaceTable[layer.top]?
                    inplaceTable[layer.top] = []
                inplaceTable[layer.top].push node
                continue
            else
                node.addChildren getNodes(layer.top, [layer.name])
        if layer.bottom?
            node.addParents getNodes(layer.bottom, [].concat layer.top)
    # Splice in the inplace nodes.
    for own k, inplaceOps of inplaceTable
        curNode = nodeTable[k]
        curNode.coalesce = inplaceOps
        children = curNode.detachChildren()
        for inplaceChild in inplaceOps
            inplaceChild.annotation = 'InPlace'
            curNode.addChild inplaceChild
            curNode = inplaceChild
        curNode.addChildren children
    # Patch in data layer parameters.
    if header?.input? and header?.input_dim?
        inputs = [].concat header.input
        dims = header.input_dim
        if inputs.length==(dims.length*0.25)
            for input, i in inputs
                dataNode = nodeTable[input]
                dataNode.type = 'data'
                dataNode.attribs.shape = dims.slice i*4, (i+1)*4
        else
            console.log 'Inconsistent input dimensions.'
    return net

module.exports =
class CaffeParser
    @parse : (txt, phase) ->
        [header, layerDesc] = Parser.parse txt
        layers = generateLayers layerDesc, phase
        network = generateNetwork layers, header
        network = analyzeNetwork network
        return network
